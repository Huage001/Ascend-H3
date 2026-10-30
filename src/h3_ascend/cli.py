"""One CLI for the four workloads, five attention paths, BF16 and INT8."""

import argparse
import json
import os
import sys
from pathlib import Path

METHODS = ("full", "vsa", "veda", "vdn", "spark")


def command(a):
    if a.parallel == "ulysses" and a.method == "vdn":
        raise ValueError("upstream VDN requires native TP head sharding; use --parallel tp")
    if a.devices not in (1, 2, 4, 7, 8):
        raise ValueError("device count must divide the 56 attention heads")
    backend = {
        "full": "TORCH_SDPA",
        "vsa": "FASTVIDEO_VSA",
        "veda": "TORCH_SDPA",
        "vdn": "VDNH3_ATTN",
        "spark": "TORCH_SDPA",
    }[a.method]
    # Identical policy for baseline and all students: current block plus one
    # asynchronously prefetched block, independent of weight dtype/method.
    # No method-dependent resident-layer count in comparative measurements.
    layer_options = {
        "dit": {"weight_transfer": "rank-local"},
        "text_encoder": {"weight_transfer": "rank-local"},
    }
    offload = [
        "--diffusion-offload-config",
        json.dumps(
            {"mode": "layer", "components": ["dit", "text_encoder"], "layer_options": layer_options}
        ),
    ]
    tp = a.devices if a.parallel == "tp" else 1
    usp = a.devices if a.parallel == "ulysses" else 1
    encoder_tp = a.devices if a.devices in (1, 2, 4, 8) else 1
    cmd = [
        sys.executable,
        "-m",
        ("h3_ascend.integrations.cuda_launch"
         if os.getenv("H3_ASCEND_CUDA_REFERENCE") == "1"
         else "vllm_omni.entrypoints.cli.main"),
        "serve",
        a.model,
        "--omni",
        "--host",
        a.host,
        "--port",
        str(a.port),
        "--task-type",
        "fl2va",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--num-gpus",
        str(a.devices),
        "--tensor-parallel-size",
        str(tp),
        "--usp",
        str(usp),
        "--ring",
        "1",
        "--text-encoder-tp-size",
        str(encoder_tp),
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        "--vae-patch-parallel-size",
        "1",
        "--enforce-eager",
        "--diffusion-attention-backend",
        backend,
        *offload,
    ]
    if os.getenv("H3_ASCEND_CUDA_REFERENCE") == "1" and a.method == "vsa":
        index = cmd.index("--diffusion-attention-backend")
        del cmd[index:index + 2]
        cmd += ["--diffusion-attention-config", json.dumps({"default": {
            "backend": "FASTVIDEO_VSA", "fastvideo_vsa_provider": "triton",
            "fastvideo_vsa_precision": "bf16",
        }})]
    if a.precision == "int8":
        cmd += [
            "--diffusion-quantization-config",
            json.dumps({"transformer": {"method": "int8"}}),
        ]
    from h3_ascend.profiles import launch_adapter

    lora = launch_adapter(a.method, getattr(a, "few_step_lora", True), a.lora)
    if lora:
        cmd += ["--lora-path", lora]
    if a.method == "vsa" and not a.lora:
        raise ValueError(
            "VSA needs --lora with trained compression gates (or prepare a fused VSA checkpoint)"
        )
    if a.method == "vdn" and not a.lora:
        raise ValueError("VDN needs --lora pointing to its stage-dmd-step-250 checkpoint directory")
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build")
    b.add_argument(
        "--method",
        choices=["all", "spark", "vsa", "veda", "vdn", "full"],
        default="all",
    )
    b.add_argument("--soc", default="Ascend910B3")
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--jobs", type=int, default=4)
    s = sub.add_parser("serve")
    s.add_argument("--method", choices=METHODS, required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--lora", help="Official VSA/VDN student, or a prepared community adapter")
    s.add_argument("--few-step-lora", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--few-step-source", type=Path, help="Local pinned LightX2V file; otherwise download")
    s.add_argument("--precision", choices=["bf16", "int8"], default="bf16")
    s.add_argument("--devices", type=int, default=1)
    s.add_argument("--parallel", choices=["tp", "ulysses"], default="tp")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--predictor")
    s.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("generate")
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--case", choices=["672-124", "672-345", "1344-124", "1344-345"], required=True)
    r.add_argument("--prompt-file", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--steps", type=int, help="Override the profile step count")
    r.add_argument("--method", choices=METHODS, default="full")
    r.add_argument("--few-step-lora", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--guidance-scale", type=float, help="Omit to retain the model's native policy")
    legacy = sub.add_parser(
        "prepare-lora", help="Prepare an H3 adapter for load-time fusion"
    )
    legacy.add_argument("--source", type=Path, required=True)
    legacy.add_argument("--output", type=Path, required=True)
    legacy.add_argument(
        "--source-format", choices=["diffusers-scaled", "raw-h3", "peft"], default="diffusers-scaled"
    )
    legacy.add_argument("--steps", type=int, help="Pin a distilled schedule; omit for a general LoRA")
    legacy.add_argument("--video-shift", type=float, default=12.0)
    legacy.add_argument("--audio-shift", type=float, default=3.0)
    a = p.parse_args()
    if a.action == "prepare-lora":
        from h3_ascend.integrations.lora import prepare

        prepare(a.source, a.output, a.steps, a.video_shift, a.audio_shift, a.source_format)
        return
    if a.action == "build":
        from h3_ascend.build import build

        build(a.method, a.soc, a.workers, a.jobs)
        return
    if a.action == "serve":
        cmd = command(a)
        if a.dry_run:
            print(
                json.dumps(
                    {"environment": {"H3_ASCEND_METHOD": a.method}, "argv": cmd},
                    indent=2,
                )
            )
            return
        from h3_ascend.assets import validate_assets
        from h3_ascend.profiles import launch_adapter

        # Preparation/merge happens before server readiness and every timed request.
        prepared = launch_adapter(
            a.method, a.few_step_lora, a.lora, prepare_assets=True, source=a.few_step_source
        )
        if prepared:
            cmd[cmd.index("--lora-path") + 1] = prepared
        validate_assets(a.method, a.lora, a.predictor)
        os.environ["H3_ASCEND_METHOD"] = a.method
        if a.predictor:
            os.environ["H3_ASCEND_VEDA_PREDICTOR"] = a.predictor
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_OMNI_VIDEO_SYNC_TIMEOUT", "7200")
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
        os.execv(sys.executable, cmd)
    if a.action == "generate":
        from h3_ascend.client import generate
        from h3_ascend.profiles import step_count

        if a.steps is None:
            a.steps = step_count(a.method, a.few_step_lora)
        generate(a)


if __name__ == "__main__":
    main()
