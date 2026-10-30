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


def validate_fasth3_v2(model):
    """Reject the legacy v1 adapter and incomplete full-checkpoint exports."""
    if not model:
        raise ValueError("VSA requires --model pointing to the complete official FastH3 V2 checkpoint")
    root = Path(model)
    contract_path = root / "fastvideo_inference.json"
    if not contract_path.is_file():
        raise ValueError("VSA requires a local official FastH3 V2 checkpoint with fastvideo_inference.json")
    contract = json.loads(contract_path.read_text())
    required = {
        "model_id": "FastVideo/FastVideo-FastH3-8-Step-V2",
        "schema_version": "fasth3-inference-contract-v1",
        "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
        "transformer_forwards": 8,
        "num_inference_steps": 9,
        "video_scheduler_shift": 10.0, "audio_scheduler_shift": 3.0,
        "guidance_scale": 1.0, "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "vsa_sparsity": 0.8, "vsa_tile_size": 64, "task": "t2av",
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(f"FastH3 V2 contract mismatch: {key}={contract.get(key)!r}; expected {expected!r}")
    if not (root / "modular_model_index.json").is_file():
        raise ValueError("FastH3 V2 requires the official modular_model_index.json")
    transformer = root / "transformer"
    indexes = list(transformer.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValueError("FastH3 V2 requires one complete transformer safetensors index")
    weight_map = json.loads(indexes[0].read_text())["weight_map"]
    gates = {
        int(m[1]) for key in weight_map
        if (m := re.fullmatch(r"transformer_blocks\.(\d+)\.attn\.to_gate_compress\.weight", key))
    }
    if gates != set(range(50)):
        raise ValueError("FastH3 V2 requires all 50 trained compression gates")
    for name in set(weight_map.values()):
        file = transformer / name
        if Path(name).name != name or not file.is_file() or file.stat().st_size < 16:
            raise ValueError(f"FastH3 V2 transformer shard missing or invalid: {name}")


def validate_assets(method, lora=None, predictor=None, *, model=None):
    if method == "veda":
        predictor = predictor or os.environ.get("H3_ASCEND_VEDA_PREDICTOR")
        if not predictor or not Path(predictor).is_file():
            raise ValueError("Veda requires its official trained predictor: pass --predictor FILE")
    if method == "vsa":
        if lora:
            raise ValueError("VSA uses the full official FastH3 V2 student; additional or legacy LoRAs are unsupported")
        validate_fasth3_v2(model)
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
