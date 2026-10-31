"""CPU checks for launch semantics, native geometry and head boundaries."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h3_ascend.cli import command, main
from h3_ascend.client import multipart, request_fields
from h3_ascend.context import HeadShard, VideoLayout
from h3_ascend.profiles import launch_adapter, step_count


class Contracts(unittest.TestCase):
    def test_cuda_reference_preloads_official_fa4_before_omni(self):
        with patch.dict("os.environ", {"H3_ASCEND_CUDA_REFERENCE": "1"}):
            for method in ("full", "vsa", "veda", "spark"):
                args = argparse.Namespace(method=method, parallel="tp", devices=1,
                    model="model", host="127.0.0.1", port=8000, precision="bf16",
                    lora=None if method == "vsa" else "checkpoint")
                argv = command(args)
                self.assertEqual(argv[2], "h3_ascend.integrations.cuda_launch")
                if method == "vsa":
                    config = json.loads(argv[argv.index("--diffusion-attention-config") + 1])
                    self.assertEqual(config["default"]["fastvideo_vsa_provider"], "triton")
                    self.assertEqual(config["default"]["fastvideo_vsa_precision"], "bf16")
                    self.assertEqual(config["per_role"]["minimax_h3.token_refiner"]["backend"],
                                     "FLASH_ATTN")

    def test_student_contracts(self):
        for method in ("full", "veda", "spark"):
            self.assertEqual(step_count(method), 8)
            self.assertEqual(step_count(method, False), 50)
            self.assertEqual(launch_adapter(method, False, None), None)
        self.assertEqual(step_count("vsa"), 8)
        self.assertEqual(step_count("vdn"), 8)
        for method in ("vsa", "vdn"):
            with self.assertRaises(ValueError):
                step_count(method, False)

    def test_v2_request_pins_trained_steps_and_guidance(self):
        argv = ["h3-ascend", "generate", "--method", "vsa", "--case", "1344-345",
                "--prompt-file", "prompt.txt", "--output", "v2.mp4"]
        with patch("sys.argv", argv), patch("h3_ascend.client.generate") as generate:
            main()
            args = generate.call_args.args[0]
            self.assertEqual((args.steps, args.guidance_scale), (8, 1.0))
        for options in (["--steps", "4"], ["--guidance-scale", "2"]):
            with patch("sys.argv", argv + options), self.assertRaises(ValueError):
                main()

    def test_native_frame_requests(self):
        for case, size, frames in [
            ("672-124", (672, 384), 124),
            ("672-345", (672, 384), 345),
            ("1344-124", (1344, 768), 124),
            ("1344-345", (1344, 768), 345),
        ]:
            fields = request_fields(case, "A moving camera.")
            self.assertEqual((fields["width"], fields["height"]), size)
            self.assertEqual(fields["num_frames"], frames)
            self.assertEqual(fields["aspect_ratio"], "16:9")
            self.assertEqual(json.loads(fields["extra_params"])["task"], "t2va")
            self.assertNotIn("seconds", fields)

    def test_head_partition_is_global(self):
        heads = [HeadShard(rank * 14, 14) for rank in range(4)]
        self.assertEqual([h.start for h in heads], [0, 14, 28, 42])
        with self.assertRaises(ValueError):
            HeadShard(42, 28)

    def test_guidance_retains_model_policy_unless_explicit(self):
        self.assertNotIn("guidance_scale", request_fields("672-124", "A park."))
        fields = request_fields("672-124", "A park.", guidance_scale=1.0)
        self.assertEqual(fields["guidance_scale"], 1.0)

    def test_layout_rejects_bad_prefix(self):
        with self.assertRaises(ValueError):
            VideoLayout((37, 12, 21), 7, prefix_segments=(3, 2))

    def test_int8_uses_native_quantization(self):
        for method in ("full", "vsa", "veda", "vdn", "spark"):
            args = argparse.Namespace(
                method=method,
                parallel="tp",
                devices=4,
                model="model",
                host="127.0.0.1",
                port=8000,
                precision="int8",
                lora=None if method == "vsa" else "checkpoint",
            )
            argv = command(args)
            self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "4")
            self.assertEqual(argv[argv.index("--usp") + 1], "1")
            self.assertIn('"int8"', argv[argv.index("--diffusion-quantization-config") + 1])
            if method in ("vsa", "vdn"):
                offload = json.loads(argv[argv.index("--diffusion-offload-config") + 1])
                self.assertNotIn("resident_layers", offload["layer_options"]["dit"])
            else:
                offload = json.loads(argv[argv.index("--diffusion-offload-config") + 1])
                self.assertNotIn("resident_layers", offload["layer_options"]["dit"])
            self.assertIn("text_encoder", offload["components"])

    def test_prompt_cannot_escape_multipart_field(self):
        body, content_type = multipart({"prompt": "a\r\nb, 中文", "num_frames": 345})
        boundary = content_type.split("boundary=")[1]
        self.assertEqual(body.count(("--" + boundary).encode()), 3)
        self.assertIn("中文".encode(), body)

    def test_marked_legacy_adapter_retains_load_time_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "h3_ascend_adapter.json").write_text("{}")
            args = argparse.Namespace(
                method="spark",
                parallel="tp",
                devices=4,
                model="model",
                host="127.0.0.1",
                port=8000,
                precision="int8",
                lora=directory,
            )
            argv = command(args)
            policy = json.loads(argv[argv.index("--diffusion-offload-config") + 1])
            self.assertNotIn("resident_layers", policy["layer_options"]["dit"])

    def test_seven_way_dit_preserves_valid_encoder_sharding(self):
        args = argparse.Namespace(
            method="full",
            parallel="tp",
            devices=7,
            model="model",
            host="127.0.0.1",
            port=8000,
            precision="bf16",
            lora=None,
        )
        argv = command(args)
        self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "7")
        self.assertEqual(argv[argv.index("--text-encoder-tp-size") + 1], "1")


if __name__ == "__main__":
    unittest.main()
