"""Native-W8A8 input exchange; original attention cores and BF16 reverse exchange."""

import json
import os
from pathlib import Path

from h3_ascend.integrations.input_projection_comm import gather_input, project, shard_weight


def install():
    if os.getenv("H3_ASCEND_INPUT_EXCHANGE") != "1":
        return
    if os.getenv("H3_ASCEND_METHOD") not in ("full", "veda", "spark", "vsa", "vdn"):
        raise ValueError("Unsupported H3 attention method")
    import torch
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel,
        _build_rope_table,
    )
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
    from h3_ascend.integrations.sequence_communication import (
        ExchangeContext,
        exchange_output,
        exchange_qkv,
    )

    if getattr(MiniMaxH3Attention, "_h3_input_exchange_installed", False):
        return
    original_load = DiffusionModelRunner.load_model
    original_projection = MiniMaxH3Attention._project_qkv
    original_rope_prepare = MiniMaxH3DiTModel.prepare_rope_table

    def prepare_rope(self, img_position_ids, *, seq_len):
        local = original_rope_prepare(self, img_position_ids, seq_len=seq_len)
        state = getattr(self, "_h3_input_rope_state", None)
        if state is None:
            return local
        # The native denoise branch calls this once per request, before its steps.
        # Every rank already owns the complete immutable position IDs.
        full = _build_rope_table(self.rope(img_position_ids).to(img_position_ids.device))
        start, length = self._rope_local_span(seq_len)
        if not torch.equal(local, full.narrow(0, start, length)):
            raise RuntimeError(
                "Rank-local slice of the cached full RoPE differs from native preparation"
            )
        state["local_rope"] = local
        state["full_rope"] = full
        state["position_ids"] = img_position_ids
        state["rope_builds"] += 1
        return local

    class InputStrategy:
        name = "ulysses"
        enabled = True

        def __init__(self, original):
            self.original = original
            self.group = original._sp_group

        def pre_attention(self, query, key, value, attn_metadata, defer_joint=False):
            if query.shape[2] == 56:
                return self.original.pre_attention(query, key, value, attn_metadata, defer_joint)
            if defer_joint:
                raise ValueError("Deferred joint tensors are not qualified")
            if any(
                getattr(attn_metadata, name, None) is not None
                for name in ("joint_query", "joint_key", "joint_value", "joint_attn_mask")
            ):
                raise ValueError("Joint/reference attention is not qualified")
            mask = getattr(attn_metadata, "attn_mask", None)
            if mask is not None and (mask.ndim != 2 or mask.shape[-1] != key.shape[1]):
                raise ValueError("Only original global packed key masks are qualified")
            if query.shape[2] != 56 // self.group.ulysses_world_size:
                raise ValueError("QKV was not head partitioned before attention")
            gate = attn_metadata.extra.get("gate_compress") if attn_metadata is not None else None
            if gate is not None:
                attn_metadata.extra["gate_compress"] = exchange_qkv(
                    (gate,), self.group.ulysses_group, self.group.ulysses_world_size, mode="auto"
                )[0]
            return (
                query,
                key,
                value,
                attn_metadata,
                ExchangeContext(self.group.ulysses_group, self.group.ulysses_world_size),
            )

        def post_attention(self, value, context):
            return exchange_output(value, context.group, context.world_size)

    @torch.compiler.disable
    def projection(self, x, rope_table, vdn_window=None):
        state = getattr(self, "_h3_input_exchange_state", None)
        if state is None:
            return original_projection(self, x, rope_table, vdn_window)
        if self.attention._get_active_parallel_strategy() is not self.attention.parallel_strategy:
            return original_projection(self, x, rope_table, vdn_window)
        if x.shape[0] < 256:
            return original_projection(self, x, rope_table, vdn_window)
        if self.vdn is not None and vdn_window is None:
            raise ValueError("Official VDN branch requires its unchanged global window")
        group = self.attention.parallel_strategy.group
        world = group.ulysses_world_size
        if x.shape[1] != 5376 or x.dtype != torch.bfloat16:
            raise ValueError("Only native BF16 input to W8A8 QKV is qualified")
        quantized, scales = gather_input(
            x.contiguous(),
            group.ulysses_group,
            world,
            packet=os.getenv("H3_ASCEND_INPUT_PACKET", "separate"),
        )
        q, k, v = project(
            quantized, scales, self._h3_input_weight, self._h3_input_scale, 56 // world
        )
        q_raw, k_raw = q, k
        del quantized, scales
        if rope_table is None:
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            full_rope, local_rope = state["full_rope"], state["local_rope"]
            if full_rope is None or full_rope.shape[0] != q.shape[0]:
                raise ValueError("Missing per-rank full RoPE cache for the current request layout")
            if rope_table.data_ptr() != local_rope.data_ptr():
                if not torch.equal(rope_table, local_rope):
                    raise ValueError("Prepared local RoPE does not match the current full cache")
                state["local_rope"] = rope_table
            q, k = fused_qk_norm_rope(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                full_rope,
                self.q_norm.variance_epsilon,
            )
        self._h3_input_calls += 1
        if not self._h3_input_audited:
            reference = original_projection(self, x, rope_table, vdn_window)
            expected = exchange_qkv(
                tuple(t[None] for t in reference[:3]), group.ulysses_group, world, mode="auto"
            )
            equal = [torch.equal(a, b[0]) for a, b in zip((q, k, v), expected)]
            raw_equal = None
            if self.vdn is not None:
                raw_reference = exchange_qkv(
                    tuple(t[None] for t in (reference[3], reference[4], reference[2])),
                    group.ulysses_group,
                    world,
                    mode="auto",
                )
                raw_equal = [torch.equal(a, b[0]) for a, b in zip((q_raw, k_raw, v), raw_reference)]
                del raw_reference
            row = dict(
                pid=os.getpid(),
                layer=self._prefix,
                forward_index=state["forward"],
                world=world,
                original_merged_quantized_weights=True,
                normalized_rope_qkv_bitexact=equal,
                vdn_raw_qkv_bitexact=raw_equal,
                local_tokens=x.shape[0],
                global_tokens=q.shape[0],
                local_heads=q.shape[1],
                added_lossy_quantization=False,
            )
            directory = Path(os.environ["H3_ASCEND_AUDIT_DIR"]).parent / "input-exchange-audit"
            directory.mkdir(exist_ok=True)
            with (directory / f"rank-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if not all(equal) or (raw_equal is not None and not all(raw_equal)):
                raise RuntimeError(
                    "Pre-QKV input exchange failed bitwise live-model QKV audit: " + self._prefix
                )
            self._h3_input_audit_steps += 1
            self._h3_input_audited = self._h3_input_audit_steps == 8
            del reference, expected
        return q, k, v, q_raw, k_raw

    def loaded(runner, *args, **kwargs):
        result = original_load(runner, *args, **kwargs)
        if not isinstance(runner.pipeline, MiniMaxH3Pipeline):
            return result
        model = runner.pipeline.transformer
        if get_tensor_model_parallel_world_size() != 1 or len(model.blocks) != 50:
            raise ValueError("Only TP1 / fifty main DiT blocks is qualified")
        strategy = model.blocks[0].attn.attention.parallel_strategy
        if not isinstance(strategy, UlyssesParallelAttention):
            return result
        # Eight-card measurements show no benefit; retain the original strategy.
        if strategy._sp_group.ulysses_world_size not in (2, 4):
            return result
        if os.getenv("H3_ASCEND_METHOD") == "vdn":
            from h3_ascend.integrations.input_projection_vdn import (
                install as install_preprojected_vdn,
            )

            install_preprojected_vdn()
        state = dict(
            audit_requested=bool(os.getenv("H3_ASCEND_AUDIT_DIR")),
            full_rope=None,
            local_rope=None,
            position_ids=None,
            forward=0,
            rope_gathers=0,
            rope_builds=0,
        )
        model._h3_input_rope_state = state
        prepared = []
        for block in model.blocks:
            attention = block.attn
            linear = attention.qkv_proj
            strategy = attention.attention.parallel_strategy
            if not isinstance(strategy, UlyssesParallelAttention):
                raise ValueError("Only pure Ulysses is qualified")
            group = strategy._sp_group
            if (
                group.ring_world_size != 1
                or group.allgather_world_size != 1
                or group.ulysses_world_size not in (2, 4, 8)
            ):
                raise ValueError("Unsupported SP layout")
            if linear.weight.dtype != torch.int8 or getattr(linear, "bias", None) is not None:
                raise ValueError("Expected native bias-free INT8 QKV after LoRA merge")
            if any(p.device.type != "npu" for p in linear.parameters()):
                raise ValueError("Initial input exchange qualification requires resident DiT")
            weight, scale = shard_weight(
                linear.weight, linear.weight_scale, group.ulysses_rank, group.ulysses_world_size
            )
            attention._h3_input_weight, attention._h3_input_scale = weight, scale
            attention._h3_input_exchange_state = state
            attention._h3_input_calls = 0
            attention._h3_input_audit_steps = 0
            attention._h3_input_audited = not state["audit_requested"]
            attention.attention.parallel_strategy = InputStrategy(strategy)
            prepared.append(
                dict(
                    layer=attention._prefix,
                    weight_shape=list(weight.shape),
                    scale_shape=list(scale.shape),
                )
            )

        def begin(module, inputs):
            state["forward"] += 1

        def finish(module, inputs, output):
            directory = os.getenv("H3_ASCEND_INPUT_RECEIPTS")
            profile = os.getenv("H3_ASCEND_PROFILE_CONTROL")
            record = state["forward"] <= 8 or (profile and (Path(profile) / "start").exists())
            if directory and record:
                path = Path(directory)
                path.mkdir(parents=True, exist_ok=True)
                receipt = dict(
                    pid=os.getpid(),
                    forward=state["forward"],
                    rope_gathers=state["rope_gathers"],
                    rope_builds=state["rope_builds"],
                    full_rope_per_rank_cached=state["full_rope"] is not None,
                    projection_calls=[block.attn._h3_input_calls for block in model.blocks],
                    all_layers_live_qkv_bitexact=(
                        all(block.attn._h3_input_audit_steps == 8 for block in model.blocks)
                        if state["audit_requested"] else None
                    ),
                    original_full_qkv_masters_retained=True,
                    prepared=prepared,
                )
                temporary = path / f"runtime-{os.getpid()}.json.new"
                temporary.write_text(json.dumps(receipt, indent=2) + "\n")
                temporary.replace(path / f"runtime-{os.getpid()}.json")

        model.register_forward_pre_hook(begin)
        model.register_forward_hook(finish, always_call=True)
        return result

    MiniMaxH3Attention._project_qkv = projection
    MiniMaxH3DiTModel.prepare_rope_table = prepare_rope
    MiniMaxH3Attention._h3_input_exchange_installed = True
    DiffusionModelRunner.load_model = loaded
