"""Load-time fusion for the explicitly marked, already-scaled legacy export.

The legacy 363-target export includes AdaLN adapters. It is distinct from the
raw LightX2V format handled by the native dynamic LoRA loader.
"""

import hashlib
import json
import math
from pathlib import Path

MANIFEST = "h3_ascend_adapter.json"
SCHEMA = "scaled-diffusers-h3-lora-v1"


def prepare(source, output, steps=8, video_shift=12.0, audio_shift=3.0):
    source = source.resolve(strict=True)
    if steps <= 0 or min(video_shift, audio_shift) <= 0:
        raise ValueError("steps and modality shifts must be positive")
    from safetensors import safe_open

    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        if not keys or any(not key.endswith((".lora_A.weight", ".lora_B.weight")) for key in keys):
            raise ValueError(
                "Expected the already-scaled legacy Diffusers export, not a raw Turbo file"
            )
    output.mkdir(parents=True, exist_ok=False)
    (output / "adapter_model.safetensors").symlink_to(source)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    config = {
        "schema": SCHEMA,
        "weight_file": "adapter_model.safetensors",
        "sha256": digest,
        "scale_already_applied": True,
        "steps": steps,
        "video_shift": video_shift,
        "audio_shift": audio_shift,
        "base_schedule": [1.0 - i / steps for i in range(steps + 1)],
    }
    (output / MANIFEST).write_text(json.dumps(config, indent=2))
    print(output.resolve())


def load_marked(path, transformer):
    root = Path(path)
    marker = root / MANIFEST
    if not marker.is_file():
        return None
    config = json.loads(marker.read_text())
    if config.get("schema") != SCHEMA or config.get("scale_already_applied") is not True:
        raise ValueError("Unsupported legacy adapter manifest")
    weights = root / config["weight_file"]
    with weights.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != config["sha256"]:
            raise ValueError("Legacy adapter bytes differ from the manifest")
    from safetensors import safe_open
    from vllm_omni.diffusion.models.minimax_h3.fasth3 import (
        FastH3WeightFusion,
        _check_block_coverage,
        _ParamPatch,
        _resolve_native_target,
    )
    from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy
    from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule

    pairs = {}
    with safe_open(weights, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            raw = name.removeprefix("transformer.")
            suffix = next(
                (s for s in (".lora_A.weight", ".lora_B.weight") if raw.endswith(s)), None
            )
            if suffix is None:
                raise ValueError(f"Unconsumed legacy adapter tensor {name}")
            value = checkpoint.get_tensor(name)
            if value.ndim != 2:
                raise ValueError(f"Legacy LoRA matrix expected: {name}")
            pairs.setdefault(raw.removesuffix(suffix), {})[suffix[6]] = value
    patches = {}
    coverage = {}
    for module, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Incomplete legacy adapter pair {module}")
        target = _resolve_native_target(module)
        if target is None:
            raise ValueError(f"Unsupported legacy adapter target {module}")
        native, layout, block = target
        if block is not None:
            coverage.setdefault(block[0], set()).add(block[1])
        patch = patches.setdefault(native + ".weight", _ParamPatch(layout=layout))
        patch.low_rank[layout] = (pair["A"], pair["B"])
    arch = transformer.arch
    _check_block_coverage(
        coverage,
        expected={
            "blocks.": arch.num_layers,
            "token_refiner.blocks.": arch.token_refiner_num_layers,
        },
        weights_path=weights,
    )
    schedule = DMD2SigmaSchedule.from_positions(tuple(config["base_schedule"]))
    if schedule.num_inference_steps != config["steps"]:
        raise ValueError("Legacy schedule does not match its step count")

    class LegacyFusion(FastH3WeightFusion):
        @property
        def base_schedule(self):
            return schedule.base_schedule

        def check_serving_contract(self, *, partition, od_config, video_shift, audio_shift):
            if partition != "fl2va":
                raise ValueError("This legacy export serves the FL2VA partition")
            if resolve_offload_strategy(od_config) is OffloadStrategy.DISTRIBUTED_LAYER_WISE:
                raise ValueError("Legacy fusion requires ordinary layerwise offload")
            if (video_shift, audio_shift) != (config["video_shift"], config["audio_shift"]):
                raise ValueError(
                    "Base model shifts differ from the explicit legacy adapter contract"
                )

        def check_task(self, task):
            if task != "t2va":
                raise ValueError("This legacy export is qualified for T2VA")

        def check_request(self, sampling, *, video_shift, audio_shift):
            if sampling.lora_request is not None or sampling.num_inference_steps != config["steps"]:
                raise ValueError(
                    f"Use the fused legacy adapter with {config['steps']} steps and no request LoRA"
                )
            extra = sampling.extra_args or {}
            for key, wanted in (("flow_shift", video_shift), ("audio_flow_shift", audio_shift)):
                if not math.isclose(float(extra.get(key, wanted)), wanted):
                    raise ValueError(f"Legacy adapter requires {key}={wanted}")

    return LegacyFusion(
        source=weights, patches=patches, head_dim=arch.attention_head_dim, requires_vsa=False
    )


def install():
    from vllm_omni.diffusion.models.minimax_h3 import fasth3, pipeline_minimax_h3

    original = fasth3.resolve_fasth3_fusion
    if getattr(original, "_h3_legacy_fusion", False):
        return

    def resolve(config, transformer):
        path = config.lora_path
        if isinstance(path, (tuple, list)) and len(path) == 1:
            path = path[0]
        if isinstance(path, str) and (Path(path) / MANIFEST).is_file():
            return load_marked(path, transformer)
        return original(config, transformer)

    resolve._h3_legacy_fusion = True
    fasth3.resolve_fasth3_fusion = resolve
    pipeline_minimax_h3.resolve_fasth3_fusion = resolve
