"""One CLI for the two 768p workloads, five attention paths, BF16 and INT8."""

import argparse
import json
import os
import sys
from pathlib import Path

METHODS = ("full", "vsa", "veda", "vdn", "spark")


def serve_environment(a):
    sp = getattr(a, "parallel", "tp") in ("sp", "ulysses")
    devices = getattr(a, "devices", 1)
    eligible_input = (
        sp
        and devices in (2, 4)
        and getattr(a, "precision", "bf16") == "int8"
        and a.offload in ("none", "text")
        and os.getenv("H3_ASCEND_CUDA_REFERENCE") != "1"
    )
    input_setting = os.getenv("H3_ASCEND_INPUT_EXCHANGE", "auto")
    if input_setting not in ("auto", "0", "1"):
        raise ValueError("H3_ASCEND_INPUT_EXCHANGE must be auto,0 or1")
    if input_setting == "1" and not eligible_input:
        raise ValueError("Input projection exchange requires Ascend INT8 SP2/SP4 with resident DiT")
    configured = getattr(a, "precompute_adaln", None)
    if configured is None:
        configured = (
            os.getenv("H3_ASCEND_PRECOMPUTE_ADALN") == "1"
            if "H3_ASCEND_PRECOMPUTE_ADALN" in os.environ
            else a.offload in ("none", "text")
        )
    if configured and a.offload not in ("none", "text"):
        raise ValueError(
            "Startup AdaLN preparation requires resident DiT; use --offload none/text or --no-precompute-adaln"
        )
    steps = getattr(a, "steps", None)
    if steps is not None and steps < 1:
        raise ValueError("Startup DiT steps must be positive")
    result = {
        "H3_ASCEND_METHOD": a.method,
        "H3_ASCEND_PRECOMPUTE_ADALN": "1" if configured else "0",
        "H3_ASCEND_SEQUENCE_COMM": os.getenv("H3_ASCEND_SEQUENCE_COMM", "auto" if sp else "native"),
        "H3_ASCEND_VDN_SP": "1" if sp and a.method == "vdn" and devices > 1 else "0",
        "H3_ASCEND_VDN_SP_CACHE_VIEWS": "1" if a.offload in ("none", "text") else "0",
        "H3_ASCEND_INPUT_EXCHANGE": "1" if eligible_input and input_setting != "0" else "0",
        # A 345-frame request can exceed the upstream 600-second RPC deadline.
        # Keep the native failure handling, but allow a complete long request.
        "VLLM_OMNI_DLO_DP_WAVE_TIMEOUT": os.getenv("VLLM_OMNI_DLO_DP_WAVE_TIMEOUT", "7200"),
    }
    if steps is not None:
        result["H3_ASCEND_ADALN_STEPS"] = str(steps)
    return result


def command(a):
    sp = a.parallel in ("sp", "ulysses")
    if sp and a.devices not in (1, 2, 4, 8):
        raise ValueError("SP supports one,two,four or eight devices")
    if a.devices not in (1, 2, 4, 7, 8):
        raise ValueError("device count must divide the 56 attention heads")
    backend = {
        "full": "TORCH_SDPA",
        "vsa": "FASTVIDEO_VSA",
        "veda": "TORCH_SDPA",
        "vdn": "VDNH3_ATTN",
        "spark": "TORCH_SDPA",
    }[a.method]
    # Prefer resident execution. Explicit fallbacks keep the selected transfer
    # policy visible in launch receipts and comparative measurements.
    policy = getattr(a, "offload", "layer")
    layer_options = {
        "dit": {"weight_transfer": "rank-local"},
        "text_encoder": {"weight_transfer": "rank-local"},
    }
    components = ["text_encoder"] if policy == "text" else ["dit", "text_encoder"]
    config = {"mode": "module" if policy == "module" else "layer", "components": components}
    if policy != "module":
        config["layer_options"] = {key: layer_options[key] for key in components}
    offload = (
        []
        if policy == "none"
        else [
            "--diffusion-offload-config",
            json.dumps(config),
        ]
    )
    tp = 1 if sp else a.devices
    usp = a.devices if sp else 1
    encoder_tp = a.devices if a.devices in (1, 2, 4, 8) else 1
    vae_parallel = getattr(a, "vae_parallel_size", 0) or a.devices
    if vae_parallel not in (1, a.devices):
        raise ValueError("H3 VAE parallel size must be one or the full device count")
    cmd = [
        sys.executable,
        "-m",
        (
            "h3_ascend.integrations.cuda_launch"
            if os.getenv("H3_ASCEND_CUDA_REFERENCE") == "1"
            else "vllm_omni.entrypoints.cli.main"
        ),
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
        str(vae_parallel),
        "--enforce-eager",
        "--diffusion-attention-backend",
        backend,
        *offload,
    ]
    if os.getenv("H3_ASCEND_CUDA_REFERENCE") == "1" and a.method == "vsa":
        index = cmd.index("--diffusion-attention-backend")
        del cmd[index : index + 2]
        cmd += [
            "--diffusion-attention-config",
            json.dumps(
                {
                    "default": {
                        "backend": "FASTVIDEO_VSA",
                        "fastvideo_vsa_provider": "triton",
                        "fastvideo_vsa_precision": "bf16",
                    },
                    "per_role": {"minimax_h3.token_refiner": {"backend": "FLASH_ATTN"}},
                }
            ),
        ]
    if a.precision == "int8":
        cmd += [
            "--diffusion-quantization-config",
            json.dumps({"transformer": {"method": "int8"}}),
        ]
    from h3_ascend.profiles import launch_adapter

    lora = launch_adapter(a.method, getattr(a, "few_step_lora", True), a.lora)
    if lora:
        cmd += ["--lora-path", lora]
    if a.method == "vsa" and a.lora:
        raise ValueError("VSA uses the complete official FastH3 V2 checkpoint; remove --lora")
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
    s.add_argument(
        "--lora",
        help="Official VDN student, or a prepared community adapter; VSA V2 is a full checkpoint",
    )
    s.add_argument("--few-step-lora", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument(
        "--few-step-source", type=Path, help="Local pinned LightX2V file; otherwise download"
    )
    s.add_argument("--precision", choices=["bf16", "int8"], default="bf16")
    s.add_argument("--devices", type=int, default=1)
    s.add_argument(
        "--steps",
        type=int,
        help="Startup DiT updates for AdaLN preparation; default resolves the loaded model/LoRA schedule",
    )
    s.add_argument(
        "--precompute-adaln",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prepare exact schedule-bound AdaLN outputs at startup; defaults on for resident DiT",
    )
    s.add_argument(
        "--parallel",
        type=str.lower,
        choices=["tp", "sp", "ulysses"],
        default="tp",
        help="DiT TP or sequence parallelism; ulysses is a compatible alias for sp",
    )
    s.add_argument(
        "--offload",
        choices=["none", "text", "module", "layer"],
        default="none",
        help="Keep weights resident by default; text offloads only the text encoder",
    )
    s.add_argument(
        "--vae-parallel-size",
        type=int,
        default=0,
        help="Native tiled VAE decode: default 0 uses all selected cards; 1 uses one",
    )
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--predictor")
    s.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("generate")
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument(
        "--case",
        choices=["1344-124", "1344-345"],
        default="1344-124",
        help="Native 768p workload (default: 124 frames)",
    )
    r.add_argument("--prompt-file", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--steps", type=int, help="Override the profile step count")
    r.add_argument("--method", choices=METHODS, default="full")
    r.add_argument("--few-step-lora", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--guidance-scale", type=float, help="Omit to retain the model's native policy")
    legacy = sub.add_parser("prepare-lora", help="Prepare an H3 adapter for load-time fusion")
    legacy.add_argument("--source", type=Path, required=True)
    legacy.add_argument("--output", type=Path, required=True)
    legacy.add_argument(
        "--source-format",
        choices=["diffusers-scaled", "raw-h3", "peft"],
        default="diffusers-scaled",
    )
    legacy.add_argument(
        "--steps", type=int, help="Pin a distilled schedule; omit for a general LoRA"
    )
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
        environment = serve_environment(a)
        if a.dry_run:
            print(
                json.dumps(
                    {"environment": environment, "argv": cmd},
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
        validate_assets(a.method, a.lora, a.predictor, model=a.model)
        os.environ.update(environment)
        if a.steps is None:
            os.environ.pop("H3_ASCEND_ADALN_STEPS", None)
        if a.method in ("vsa", "vdn") and a.precision == "bf16":
            # File-backed input-only vision masters keep long T2VA requests
            # below host-memory limits without changing active model math.
            os.environ.setdefault("H3_ASCEND_DORMANT_VISION_MMAP", "1")
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
        if a.method == "vsa" and a.steps != 8:
            raise ValueError("FastH3 V2 requires eight forwards with its trained sigma schedule")
        if a.method == "vsa":
            if a.guidance_scale not in (None, 1.0):
                raise ValueError("FastH3 V2 requires guidance_scale=1")
            a.guidance_scale = 1.0
        generate(a)


if __name__ == "__main__":
    main()
