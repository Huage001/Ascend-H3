"""Method-specific asset contracts; no model-name whitelist for dense H3 LoRAs."""

import json
import os
import re
from pathlib import Path

VDN_TRANSFORMS = [{
    "type": "hybrid_attention", "version": 2,
    "config": {
        "anchor_frames": "both", "enable_softmax_gate": True,
        "softmax_attention": {"chunk": 5, "radius": 1},
        "linear_attention": {
            "a_fp32": True, "bridge": "alpha", "delta_rule": "vdn_solve",
            "enable_text_state": True, "linear_head_dim": 128,
            "short_conv": {"targets": ["k", "v"]},
        },
    },
}]


def adapter_file(path):
    if not path:
        return None
    p = Path(path)
    if p.is_dir():
        p = p / "adapter_model.safetensors"
    return p if p.is_file() else None


def adapter_header(path):
    """Inspect only the standard safetensors JSON header, without importing Torch."""
    p = adapter_file(path)
    if p is None:
        raise ValueError(f"Local adapter file not found: {path}")
    with p.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if not 0 < size <= min(64 * 1024 * 1024, p.stat().st_size - 8):
            raise ValueError(f"Invalid safetensors header: {p}")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid adapter header: {p}")
    return header


def is_fasth3(path):
    p = adapter_file(path)
    if p is None:
        return False
    metadata = adapter_header(p).get("__metadata__", {})
    return metadata.get("format") == "fastvideo-lora-v2" and metadata.get(
        "finetuned_model", ""
    ).startswith(("FastVideo/FastH3", "FastVideo/FastVideo-FastH3"))


def validate_assets(method, lora=None, predictor=None):
    if method == "veda":
        predictor = predictor or os.environ.get("H3_ASCEND_VEDA_PREDICTOR")
        if not predictor or not Path(predictor).is_file():
            raise ValueError("Veda requires its official trained predictor: pass --predictor FILE")
    if method == "vsa":
        if not is_fasth3(lora):
            raise ValueError("VSA requires the official FastH3 VSA LoRA, without extra community LoRAs")
        header = adapter_header(lora)
        gates = {
            int(m[1]) for key in header
            if (m := re.fullmatch(r"transformer_blocks\.(\d+)\.attn\.to_gate_compress\.set_weight", key))
        }
        factors = {key.removesuffix(".lora_A.weight") for key in header if key.endswith(".lora_A.weight")}
        paired = {key.removesuffix(".lora_B.weight") for key in header if key.endswith(".lora_B.weight")}
        if not factors or factors != paired:
            raise ValueError("VSA needs the complete FastH3 student LoRA, not only compression gates")
        if gates != set(range(50)):
            raise ValueError("Use the FastH3 VSA variant with all 50 trained gates, not its dense variant")
    if method == "vdn":
        root = Path(lora) if lora else Path("")
        if not all((root / name).is_file() for name in (
            "model_spec.json", "metadata.json", "adapters/default/adapter_model.safetensors",
            "adapters/turbo/adapter_model.safetensors", "linear_branch/model.safetensors"
        )):
            raise ValueError("VDN requires the complete official stage-dmd-step-250 checkpoint")
        spec = json.loads((root / "model_spec.json").read_text())
        metadata = json.loads((root / "metadata.json").read_text())
        train = metadata.get("metadata", {})
        if (
            spec.get("base", {}).get("class_name") != "MiniMaxH3Transformer3DModel"
            or spec.get("transforms") != VDN_TRANSFORMS
            or metadata.get("kind") != "weights"
            or (train.get("stage"), train.get("step"), train.get("turbo_num_steps")) != ("dmd", 250, 8)
        ):
            raise ValueError("VDN supports its official released configuration only; do not stack another LoRA")
