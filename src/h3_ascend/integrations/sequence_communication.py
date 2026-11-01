"""Opt-in, exact Ulysses communication candidates; attention math is unchanged.

`batched` exchanges Q/K/V (and a compatible VSA gate) in one collective.
`async` submits separate exchanges before waiting, preserving their layout.
Every call owns its buffers: concurrent requests never reuse live output storage.
No cross-rank reduction or low-precision communication is introduced.
"""

import os
from dataclasses import dataclass

_FAILED_EXCHANGES = []


@dataclass(frozen=True)
class ExchangeContext:
    group: object
    world_size: int
    name: str = "h3_exact_sequence_exchange"


def eligible(tensors, world_size):
    """Only equal-shaped inference tensors in strict batch-one H3 layout."""
    if world_size <= 1 or not tensors:
        return False
    first = tensors[0]
    if first.ndim != 4 or first.shape[0] != 1 or first.shape[1] < 1:
        return False
    if first.shape[2] % world_size or first.shape[3] < 1:
        return False
    return all(
        tensor.shape == first.shape
        and tensor.dtype == first.dtype
        and tensor.device == first.device
        and not tensor.requires_grad
        for tensor in tensors
    )


def _exchange(send, group, dist):
    import torch

    received = torch.empty_like(send)
    # All owning tensors stay in this call until communication completes.
    try:
        work = dist.all_to_all_single(received, send, group=group, async_op=True)
    except BaseException:
        _FAILED_EXCHANGES.append((send, received, group))
        raise
    return received, work


def exchange_qkv(tensors, group, world_size, *, mode="batched", dist=None):
    """[1, local_tokens, all_heads, D] -> full sequence, local heads.

    Preconditions are rank-consistent: the pinned H3 architecture fixes dtype,
    dimensions, gate presence and inference mode for every rank in the group.
    Unsupported contracts must fall back *before* any collective is submitted.
    """
    import torch
    import torch.distributed as distributed

    dist = distributed if dist is None else dist
    if not eligible(tensors, world_size):
        raise ValueError("Only equal-shaped strict inference Q/K/V[/gate] are supported")
    if mode not in ("batched", "async", "auto"):
        raise ValueError(f"Unknown sequence communication mode {mode}")
    if mode == "auto":
        # Small messages benefit from fewer launches; the four-card checks
        # showed extra receive packing loses on the real long-video tensors.
        limit = int(os.getenv("H3_ASCEND_SEQUENCE_BATCH_MAX_BYTES", str(2 * 1024**2)))
        payload = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        mode = "batched" if payload <= limit else "async"
    _, tokens, heads, width = tensors[0].shape
    local_heads = heads // world_size

    if mode == "batched":
        # Write directly to destination-major send slices; no stack+permute
        # intermediate. The receiver converts once to component-major output.
        send = torch.empty(
            (world_size, len(tensors), tokens, local_heads, width),
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )
        for index, tensor in enumerate(tensors):
            source = tensor.reshape(1, tokens, world_size, local_heads, width)
            send[:, index].copy_(source[0].permute(1, 0, 2, 3))
        received, work = _exchange(send, group, dist)
        try:
            work.wait()
        except BaseException:
            _FAILED_EXCHANGES.append((send, received, work, group))
            raise
        del send, source
        # Splitting this allocation yields disjoint, contiguous Q/K/V buffers.
        output = received.permute(1, 0, 2, 3, 4).contiguous()
        return tuple(value.reshape(1, world_size * tokens, local_heads, width)
                     for value in output.unbind(0))

    # Submit all independent components without CPU waits between collectives.
    # Packing the next tensor may overlap a prior transfer where HCCL allows it;
    # this is an opportunity, not an assertion of measured overlap.
    pending = []
    try:
        for tensor in tensors:
            send = tensor.reshape(1, tokens, world_size, local_heads, width)
            send = send.transpose(0, 2).contiguous()
            received, work = _exchange(send, group, dist)
            pending.append((send, received, work))
        for _, _, work in pending:
            work.wait()
    except BaseException:
        _FAILED_EXCHANGES.append((tensors, pending, group))
        raise
    return tuple(received.reshape(1, world_size * tokens, local_heads, width)
                 for _, received, _ in pending)


def exchange_output(tensor, group, world_size, *, dist=None):
    """Reverse exchange with native source-rank/head ordering, batch one."""
    import torch.distributed as distributed

    dist = distributed if dist is None else dist
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] % world_size:
        raise ValueError("Reverse exchange requires batch one and divisible token count")
    if tensor.requires_grad:
        raise ValueError("This integration is inference only")
    _, global_tokens, local_heads, width = tensor.shape
    tokens = global_tokens // world_size
    # A destination's query span is already contiguous in BSND. Send it in
    # token-major order, avoiding the native [destination,head,token,D] copy.
    # The receive still concatenates source ranks in global head order.
    send = tensor.reshape(world_size, tokens, local_heads, width).contiguous()
    received, work = _exchange(send, group, dist)
    try:
        work.wait()
    except BaseException:
        _FAILED_EXCHANGES.append((tensor, send, received, work, group))
        raise
    del send
    return received.permute(1, 0, 2, 3).contiguous().reshape(
        1, tokens, world_size * local_heads, width)


def install():
    """Patch only the parallel strategy; fall back for every unsupported case."""
    mode = os.getenv("H3_ASCEND_SEQUENCE_COMM", "native")
    if mode == "native":
        return
    if mode not in ("batched", "async", "auto"):
        raise ValueError("H3_ASCEND_SEQUENCE_COMM must be native, batched, async or auto")
    from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
    from vllm_omni.diffusion.forward_context import get_ulysses_mode

    if getattr(UlyssesParallelAttention, "_h3_exact_comm", False):
        return
    original_pre = UlyssesParallelAttention.pre_attention
    original_post = UlyssesParallelAttention.post_attention

    def pre(self, query, key, value, attn_metadata, defer_joint=False):
        import torch

        group = self._sp_group
        tensors = (query, key, value)
        supported = (
            get_ulysses_mode(default="strict") == "strict"
            and self._scatter_idx == 2 and self._gather_idx == 1
            and group.ring_world_size == 1
            and getattr(group, "allgather_world_size", 1) == 1
            and not defer_joint
            and not self._ulysses_a2a_permute
            and not self._use_sync
            and not torch.is_grad_enabled()
            and eligible(tensors, group.ulysses_world_size)
            and all(getattr(attn_metadata, name, None) is None for name in
                    ("joint_query", "joint_key", "joint_value", "attn_mask", "joint_attn_mask"))
        )
        gate = attn_metadata.extra.get("gate_compress") if attn_metadata is not None else None
        if gate is not None:
            supported = supported and isinstance(gate, torch.Tensor) and eligible(
                (*tensors, gate), group.ulysses_world_size)
            tensors = (*tensors, gate)
        if not supported:
            return original_pre(self, query, key, value, attn_metadata, defer_joint)
        exchanged = exchange_qkv(tensors, self._ulysses_pg, group.ulysses_world_size, mode=mode)
        if gate is not None:
            attn_metadata.extra["gate_compress"] = exchanged[3]
        ctx = ExchangeContext(self._ulysses_pg, group.ulysses_world_size)
        return (*exchanged[:3], attn_metadata, ctx)

    def post(self, attn_output, ctx):
        if not isinstance(ctx, ExchangeContext):
            return original_post(self, attn_output, ctx)
        return exchange_output(attn_output, ctx.group, ctx.world_size)

    UlyssesParallelAttention.pre_attention = pre
    UlyssesParallelAttention.post_attention = post
    UlyssesParallelAttention._h3_exact_comm = True
