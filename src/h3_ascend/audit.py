"""Opt-in, synchronizing inference audit; exclude audited runs from timing."""

import json
import os
from pathlib import Path


def install():
    directory = os.getenv("H3_ASCEND_AUDIT_DIR")
    if not directory:
        return
    import torch
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    original = MiniMaxH3DiTModel.forward
    if getattr(original, "_h3_audited", False):
        return

    @torch.compiler.disable
    def forward(self, *args, **kwargs):
        if not hasattr(self, "_h3_audit_counts"):
            self._h3_audit_counts = [0] * len(self.blocks)
            self._h3_audit_handles = []
            for i, block in enumerate(self.blocks):

                def count(module, inputs, output, index=i):
                    self._h3_audit_counts[index] += 1

                self._h3_audit_handles.append(block.register_forward_hook(count))
        before = list(self._h3_audit_counts)
        result = original(self, *args, **kwargs)
        tensors = [value for value in result if isinstance(value, torch.Tensor)]
        finite = bool(tensors) and all(bool(torch.isfinite(value).all()) for value in tensors)
        gates = [getattr(block.attn, "to_gate_compress", None) for block in self.blocks]
        row = {
            "pid": os.getpid(),
            "method": os.environ["H3_ASCEND_METHOD"],
            "output_shapes": [list(value.shape) for value in tensors],
            "finite": finite,
            "block_forward_counts": [a - b for a, b in zip(self._h3_audit_counts, before)],
            "qkv_weight_dtypes": [
                str(getattr(block.attn.qkv_proj, "weight", None).dtype)
                if isinstance(getattr(block.attn.qkv_proj, "weight", None), torch.Tensor)
                else "nonstandard weight storage"
                for block in self.blocks
            ],
            "nonzero_vsa_gates": sum(
                bool(torch.count_nonzero(gate.weight)) for gate in gates if gate is not None
            ),
        }
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"dit-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if not finite or any(count != 1 for count in row["block_forward_counts"]):
            raise RuntimeError("DiT finite-output/block-count audit failed")
        return result

    forward._h3_audited = True
    MiniMaxH3DiTModel.forward = forward
