"""Opt-in, synchronizing inference audit; exclude audited runs from timing."""

import json
import os
import traceback
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
            from h3_ascend.profiling import register_live_model

            register_live_model(self)
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            from vllm_omni.diffusion.offloader.config import resolve_offload

            policy = resolve_offload(self.od_config)
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            hooks = []
            for block in self.blocks:
                registry = getattr(block, "_hook_registry", None)
                hook = registry.get_hook("layerwise_offload") if registry else None
                hooks.append({
                    "implementation": type(hook).__module__ + "." + type(hook).__name__,
                    "pin_memory": getattr(hook, "pin_memory", None),
                    "cpu_master_bytes": sum(
                        t.numel() * t.element_size()
                        for t in getattr(hook, "dtype_cpu_flattened_weights", {}).values()
                    ),
                })
            linears = []
            for name, module in self.named_modules():
                quant = getattr(module, "quant_method", None)
                if quant is not None:
                    weight = getattr(module, "weight", None)
                    linears.append({"name": name, "method": type(quant).__name__,
                                    "weight_dtype": str(weight.dtype) if weight is not None else None})
            from h3_ascend.integrations.adaln_precompute import runtime_stats

            adaln_stats = runtime_stats(self)
            if os.getenv("H3_ASCEND_PRECOMPUTE_ADALN") == "1":
                assert adaln_stats and adaln_stats["adaln_weights_cpu"]
                assert adaln_stats["other_dit_parameters_npu"]
            (root / f"runtime-{os.getpid()}.json").write_text(json.dumps({
                "offload_strategy": policy.strategy.value,
                "components": sorted(policy.components),
                "resident_layers": policy.resident_layers,
                "pin_memory": policy.pin_memory,
                "block_hooks": hooks,
                "linear_methods": linears,
                **__import__("h3_ascend.parallel_runtime",fromlist=["metadata"]).metadata(self),
                "tp_rank": tp_rank,
                "tp_world_size": tp_size,
                "local_attention_heads": [block.attn.num_heads for block in self.blocks],
                "total_attention_heads": [block.attn.total_num_heads for block in self.blocks],
                "adaln_precompute": adaln_stats,
            }, indent=2))
            self._h3_audit_counts = [0] * len(self.blocks)
            self._h3_audit_gates = [None] * len(self.blocks)
            self._h3_audit_handles = []
            self._h3_local_block_shapes = {}
            self._h3_global_block_tokens = {}
            for i, block in enumerate(self.blocks):

                def capture_input(module, inputs, keyword_inputs, index=i):
                    self._h3_local_block_shapes[index]=list(inputs[0].shape)
                    self._h3_global_block_tokens[index]=int(keyword_inputs['packed_total'])
                self._h3_audit_handles.append(block.register_forward_pre_hook(capture_input,with_kwargs=True))

                def count(module, inputs, output, index=i):
                    self._h3_audit_counts[index] += 1

                self._h3_audit_handles.append(block.register_forward_hook(count))
                gate = getattr(block.attn, "to_gate_compress", None)
                if gate is not None:

                    def check_gate(module, inputs, index=i):
                        # Parent block offload hooks materialize the projection
                        # before this child runs, then clear it after the block.
                        self._h3_audit_gates[index] = bool(torch.count_nonzero(module.weight))

                    self._h3_audit_handles.append(gate.register_forward_pre_hook(check_gate))
        before = list(self._h3_audit_counts)
        from h3_ascend.integrations.adaln_precompute import runtime_stats

        adaln_before = runtime_stats(self)
        self._h3_audit_gates = [None] * len(self.blocks)
        result = original(self, *args, **kwargs)
        tensors = [value for value in result if isinstance(value, torch.Tensor)]
        finite = bool(tensors) and all(bool(torch.isfinite(value).all()) for value in tensors)
        row = {
            "pid": os.getpid(),
            "method": os.environ["H3_ASCEND_METHOD"],
            "output_shapes": [list(value.shape) for value in tensors],
            "local_block_input_shapes": self._h3_local_block_shapes,
            "global_block_packed_tokens": self._h3_global_block_tokens,
            "vdn_sequence_calls": [getattr(block.attn.vdn, "_h3_vdn_sequence_calls", 0)
                                   if block.attn.vdn is not None else 0 for block in self.blocks],
            "finite": finite,
            "block_forward_counts": [a - b for a, b in zip(self._h3_audit_counts, before)],
            "qkv_weight_dtypes": [
                str(getattr(block.attn.qkv_proj, "weight", None).dtype)
                if isinstance(getattr(block.attn.qkv_proj, "weight", None), torch.Tensor)
                else "nonstandard weight storage"
                for block in self.blocks
            ],
            "nonzero_vsa_gates": sum(value is True for value in self._h3_audit_gates),
            "vsa_gate_checks": self._h3_audit_gates,
            "vsa_sparsities": [getattr(block.attn, "vsa_sparsity", None) for block in self.blocks],
            "vsa_tile_sizes": [getattr(block.attn, "vsa_tile_size", None) for block in self.blocks],
        }
        if adaln_before is not None:
            adaln_after = runtime_stats(self)
            row["adaln_cache_hits"] = adaln_after["cache_hits"] - adaln_before["cache_hits"]
            row["adaln_cache_misses"] = adaln_after["cache_misses"] - adaln_before["cache_misses"]
            row["adaln_weight_stage_loads"] = adaln_after["weight_stage_loads"] - adaln_before["weight_stage_loads"]
            assert row["adaln_cache_hits"] >= 51 and row["adaln_cache_misses"] == 0
            assert row["adaln_weight_stage_loads"] == 0
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"dit-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if not finite or any(count != 1 for count in row["block_forward_counts"]):
            raise RuntimeError("DiT finite-output/block-count audit failed")
        limit = int(os.getenv("H3_ASCEND_AUDIT_FORWARDS", "0"))
        self._h3_audit_completed = getattr(self, "_h3_audit_completed", 0) + 1
        if limit and self._h3_audit_completed >= limit:
            for handle in self._h3_audit_handles:
                handle.remove()
            self._h3_audit_handles.clear()
            MiniMaxH3DiTModel.forward = original
            (root / f"audit-detached-{os.getpid()}.json").write_text(
                json.dumps({"audited_forwards": self._h3_audit_completed, "hooks_removed": True})
            )
        return result

    @torch.compiler.disable
    def reporting_forward(self, *args, **kwargs):
        try:
            return forward(self, *args, **kwargs)
        except Exception as error:
            from h3_ascend.integrations.adaln_precompute import runtime_stats

            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"error-{os.getpid()}.json"
            path.write_text(json.dumps({
                "type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(),
                "block_counts": getattr(self, "_h3_audit_counts", None),
                "adaln_precompute": runtime_stats(self),
            }, indent=2) + "\n")
            # An empty AssertionError message is otherwise lost by the native
            # async RPC pump's truthiness check. Preserve a nonempty diagnostic.
            raise RuntimeError(f"DiT warmup audit failed: {type(error).__name__}; see {path}") from error

    reporting_forward._h3_audited = True
    MiniMaxH3DiTModel.forward = reporting_forward
