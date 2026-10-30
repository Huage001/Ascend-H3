"""Catch the gate-only student and dangling-checkpoint failures before loading."""

import json
import tempfile
import unittest
from pathlib import Path

from h3_ascend.assets import validate_assets, validate_fasth3_v2


class AssetContracts(unittest.TestCase):
    def test_gate_only_file_does_not_count_as_fasth3_student(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_model.safetensors"
            header = {"__metadata__": {"format": "fastvideo-lora-v2",
                                      "finetuned_model": "FastVideo/FastVideo-FastH3-4-step-v1"}}
            for layer in range(50):
                header[f"transformer_blocks.{layer}.attn.to_gate_compress.set_weight"] = {
                    "dtype": "F32", "shape": [1], "data_offsets": [0, 4]
                }
            encoded = json.dumps(header).encode()
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(4))
            with self.assertRaisesRegex(ValueError, "full official FastH3 V2"):
                validate_assets("vsa", directory)

    def test_v2_rejects_wrong_schedule_missing_gates_and_missing_shards(self):
        source = Path(__file__).resolve().parents[1] / "configs/fasth3-v2.json"
        spec = json.loads(source.read_text())
        contract = {
            "model_id": spec["repo_id"], "schema_version": "fasth3-inference-contract-v1",
            "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
            "transformer_forwards": 8, "num_inference_steps": 9,
            "video_scheduler_shift": 10., "audio_scheduler_shift": 3.,
            "guidance_scale": 1., "attention_backend": "VIDEO_SPARSE_ATTN_H3",
            "vsa_sparsity": .8, "vsa_tile_size": 64, "task": "t2av",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "fastvideo_inference.json"
            meta.write_text(json.dumps(dict(contract, video_scheduler_shift=12.)))
            with self.assertRaisesRegex(ValueError, "video_scheduler_shift"):
                validate_fasth3_v2(root)
            meta.write_text(json.dumps(contract))
            (root / "modular_model_index.json").write_text("{}")
            (root / "transformer").mkdir()
            index = root / "transformer/model.safetensors.index.json"
            gates = {f"transformer_blocks.{i}.attn.to_gate_compress.weight": "shard.safetensors" for i in range(50)}
            index.write_text(json.dumps({"weight_map": gates}))
            with self.assertRaisesRegex(ValueError, "shard missing"):
                validate_fasth3_v2(root)
            (root / "transformer/shard.safetensors").write_bytes(bytes(32))
            validate_fasth3_v2(root)
            del gates["transformer_blocks.49.attn.to_gate_compress.weight"]
            index.write_text(json.dumps({"weight_map": gates}))
            with self.assertRaisesRegex(ValueError, "all 50"):
                validate_fasth3_v2(root)

    def test_vdn_directories_do_not_hide_missing_weight_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_spec.json").write_text("{}")
            (root / "metadata.json").write_text("{}")
            (root / "adapters/default").mkdir(parents=True)
            (root / "adapters/turbo").mkdir(parents=True)
            (root / "linear_branch").mkdir()
            (root / "adapters/default/adapter_model.safetensors").symlink_to(root / "absent")
            with self.assertRaisesRegex(ValueError, "complete official"):
                validate_assets("vdn", directory)
