"""Load-time fusion of prepared community H3 DiT LoRAs, including partial ones."""

import hashlib
import json
import math
from pathlib import Path

MANIFEST = "h3_ascend_adapter.json"
SCHEMA = "scaled-diffusers-h3-lora-v1"


def prepare(
    source, output, steps=None, video_shift=12.0, audio_shift=3.0, source_format="diffusers-scaled"
):
    source = source.resolve(strict=True)
    if (steps is not None and steps <= 0) or min(video_shift, audio_shift) <= 0:
        raise ValueError("steps and modality shifts must be positive")
    from safetensors import safe_open

    converted = None
    if source_format == "raw-h3":
        from diffusers.loaders import MiniMaxH3LoraLoaderMixin

        converted = MiniMaxH3LoraLoaderMixin.lora_state_dict(str(source), local_files_only=True)
        keys = list(converted)
    elif source_format == "diffusers-scaled":
        with safe_open(source, framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
    elif source_format == "peft":
        import torch

        converted = {}
        with safe_open(source, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            if "alpha" not in metadata:
                raise ValueError("PEFT source must declare alpha; do not guess adapter scaling")
            alpha = float(metadata["alpha"])
            for key in checkpoint.keys():
                normalized = key.replace(".lora_A.default.weight", ".lora_A.weight").replace(
                    ".lora_B.default.weight", ".lora_B.weight"
                )
                if normalized in converted:
                    raise ValueError(f"Duplicate normalized PEFT key: {normalized}")
                converted[normalized] = checkpoint.get_tensor(key)
        for key, value in list(converted.items()):
            if key.endswith(".lora_A.weight"):
                bkey = key.removesuffix(".lora_A.weight") + ".lora_B.weight"
                if bkey not in converted or value.ndim != 2:
                    raise ValueError(f"Incomplete PEFT pair: {key}")
                b = converted[bkey]
                if b.ndim != 2 or b.shape[1] != value.shape[0]:
                    raise ValueError(f"Incompatible PEFT factor dimensions: {key}")
                scale = alpha / value.shape[0]
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError("PEFT alpha/rank must be finite and positive")
                if scale != 1.0:
                    converted[bkey] = b.to(torch.float32) * scale
        keys = list(converted)
    else:
        raise ValueError(f"Unknown adapter source format: {source_format}")
    if not keys or any(not key.endswith((".lora_A.weight", ".lora_B.weight")) for key in keys):
        raise ValueError("Expected a complete, scaled Diffusers H3 adapter after conversion")
    pairs = {}
    for key in keys:
        suffix = ".lora_A.weight" if key.endswith(".lora_A.weight") else ".lora_B.weight"
        pairs.setdefault(key.removesuffix(suffix), set()).add(suffix)
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("Every adapter target must have both A and B factors")
    output.mkdir(parents=True, exist_ok=False)
    weights = output / "adapter_model.safetensors"
    if converted is None:
        weights.symlink_to(source)
    else:
        from safetensors.torch import save_file

        save_file({key: value.contiguous().clone() for key, value in converted.items()}, weights)
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with source.open("rb") as stream:
        source_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    config = {
        "schema": SCHEMA,
        "weight_file": "adapter_model.safetensors",
        "sha256": digest,
        "scale_already_applied": True,
        "source_format": source_format,
        "source_sha256": source_digest,
        "steps": steps,
        "video_shift": video_shift if steps is not None else None,
        "audio_shift": audio_shift if steps is not None else None,
        "base_schedule": [1.0 - i / steps for i in range(steps + 1)] if steps else None,
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
    for module, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Incomplete legacy adapter pair {module}")
        target = _resolve_native_target(module)
        if target is None:
            raise ValueError(f"Unsupported legacy adapter target {module}")
        native, layout, _ = target
        patch = patches.setdefault(native + ".weight", _ParamPatch(layout=layout))
        patch.low_rank[layout] = (pair["A"], pair["B"])
    arch = transformer.arch
    schedule = (
        DMD2SigmaSchedule.from_positions(tuple(config["base_schedule"]))
        if config.get("base_schedule") is not None else None
    )
    if schedule is not None and schedule.num_inference_steps != config["steps"]:
        raise ValueError("Legacy schedule does not match its step count")

    class LegacyFusion(FastH3WeightFusion):
        @property
        def serving_flow_shifts(self):
            return (config["video_shift"], config["audio_shift"]) if schedule else None

        @property
        def base_schedule(self):
            return schedule.base_schedule if schedule is not None else None

        def fuse(self, name, weight):
            patch = self._patches.get(name)
            if patch is None or patch.layout not in ("q", "k", "v") or len(patch.low_rank) == 3:
                return super().fuse(name, weight)
            # A community adapter may modify Q alone. The full-student fuser
            # requires Q/K/V together; update only the supplied head slots.
            self._applied.add(name)
            device = self._compute_device(weight)
            result = self._widen(weight, device).clone()
            grouped = result.view(-1, 3, self._head_dim, weight.shape[-1])
            for slot, (a, b) in patch.low_rank.items():
                delta = self._widen(b, device) @ self._widen(a, device)
                grouped[:, "qkv".index(slot)].add_(delta.view_as(grouped[:, 0]))
            return result.to(weight.dtype)

        def check_serving_contract(self, *, partition, od_config, video_shift, audio_shift):
            if partition != "fl2va":
                raise ValueError("This legacy export serves the FL2VA partition")
            if resolve_offload_strategy(od_config) is OffloadStrategy.DISTRIBUTED_LAYER_WISE:
                raise ValueError("Legacy fusion requires ordinary layerwise offload")
            if schedule is not None and (video_shift, audio_shift) != (
                config["video_shift"], config["audio_shift"]
            ):
                raise ValueError(
                    "Base model shifts differ from the explicit legacy adapter contract"
                )

        def check_task(self, task):
            if task != "t2va":
                raise ValueError("This legacy export is qualified for T2VA")

        def check_request(self, sampling, *, video_shift, audio_shift):
            if sampling.lora_request is not None:
                raise ValueError("The community adapter is already fused; omit request-time LoRA")
            if schedule is not None and sampling.num_inference_steps != config["steps"]:
                raise ValueError(
                    f"Use the fused legacy adapter with {config['steps']} steps and no request LoRA"
                )
            if schedule is None:
                return
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
        from h3_ascend.profiles import STACK

        if isinstance(path, str) and (Path(path) / STACK).is_file():
            paths = json.loads((Path(path) / STACK).read_text())["adapters"]
            children = [load_marked(p, transformer) for p in paths]
            if not children or any(child is None or child.requires_vsa for child in children):
                raise ValueError("Static composition requires prepared community adapters")
            schedules = [child for child in children if child.base_schedule is not None]
            if len(schedules) > 1:
                raise ValueError("Only one distilled schedule may be composed")

            class StackFusion:
                source = Path(path)
                requires_vsa = False
                base_schedule = schedules[0].base_schedule if schedules else None
                serving_flow_shifts = schedules[0].serving_flow_shifts if schedules else None

                def apply(self, stream):
                    for child in children:
                        stream = child.apply(stream)
                    return stream

                def validate_fully_applied(self, loaded):
                    for child in children:
                        child.validate_fully_applied(loaded)

                def check_serving_contract(self, **kwargs):
                    for child in children:
                        child.check_serving_contract(**kwargs)

                def check_task(self, task):
                    for child in children:
                        child.check_task(task)

                def check_request(self, sampling, **kwargs):
                    for child in children:
                        child.check_request(sampling, **kwargs)

            return StackFusion()
        return original(config, transformer)

    resolve._h3_legacy_fusion = True
    fasth3.resolve_fasth3_fusion = resolve
    pipeline_minimax_h3.resolve_fasth3_fusion = resolve
