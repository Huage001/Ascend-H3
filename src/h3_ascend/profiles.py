"""Explicit student schedules and reproducible, pre-inference LoRA preparation."""

import hashlib
import json
from pathlib import Path

from h3_ascend.paths import cache_root, source_root

STACK = "h3_ascend_stack.json"


def step_count(method, few_step_lora=True):
    if method in ("vsa", "vdn"):
        if not few_step_lora:
            raise ValueError(f"{method} requires its own official student")
        return 4 if method == "vsa" else 8
    return 4 if few_step_lora else 50


def lightx_path():
    return cache_root() / "models/lightx2v4-prepared"


def prepare_lightx(source=None):
    """Check the official artifact and normalize PEFT scaling before loading H3."""
    from h3_ascend.integrations.lora import MANIFEST, prepare

    target = lightx_path()
    spec = json.loads((source_root() / "configs/lightx2v4.json").read_text())
    if (target / MANIFEST).is_file():
        config = json.loads((target / MANIFEST).read_text())
        if config.get("source_sha256") != spec["sha256"]:
            raise ValueError("Cached LightX2V adapter does not match the pinned artifact")
        for field in ("steps", "video_shift", "audio_shift"):
            if config.get(field) != spec[field]:
                raise ValueError(f"Cached LightX2V {field} differs from the pinned schedule")
        return target
    if source is None:
        from huggingface_hub import hf_hub_download

        source = hf_hub_download(
            repo_id=spec["repo_id"], filename=spec["filename"], revision=spec["revision"]
        )
    source = Path(source).resolve(strict=True)
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != spec["sha256"]:
            raise ValueError("LightX2V source SHA256 differs from the pinned official weights")
    prepare(source, target, spec["steps"], spec["video_shift"], spec["audio_shift"], "peft")
    return target


def launch_adapter(method, few_step_lora, lora, *, prepare_assets=False, source=None):
    step_count(method, few_step_lora)
    if method in ("vsa", "vdn") or not few_step_lora:
        return lora
    default = prepare_lightx(source) if prepare_assets else lightx_path()
    if not lora or Path(lora).resolve() == default.resolve():
        return str(default)
    # Static composition: no adapter dispatch or extra GEMM during inference.
    identity = hashlib.sha256(str(Path(lora).resolve()).encode()).hexdigest()[:20]
    stack = cache_root() / "models" / ("lightx2v4-plus-" + identity)
    if prepare_assets:
        from h3_ascend.integrations.lora import MANIFEST

        marker = Path(lora) / MANIFEST
        if not marker.is_file():
            raise ValueError("Prepare the community LoRA with prepare-lora before static composition")
        if json.loads(marker.read_text()).get("base_schedule") is not None:
            raise ValueError("Do not stack two distilled students; use --no-few-step-lora for an explicit student")
        stack.mkdir(parents=True, exist_ok=True)
        (stack / STACK).write_text(json.dumps({"adapters": [str(default), str(Path(lora).resolve())]}))
    return str(stack)
