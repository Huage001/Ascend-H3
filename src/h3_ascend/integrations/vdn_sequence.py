"""Resident, replicated-weight VDN adapter for pure Ulysses T2VA inference.

The original per-head features, delta factors, window exclusion and bidirectional
scans are called unchanged. Global x preserves the original frame-mean order.
Only head-independent parameter views and communication are added here.
"""

import copy
import os


def branch_head_view(branch, first, count, head_dim):
    """Share complete master weights; select head-local outputs without GEMM changes."""
    import torch

    class Select(torch.nn.Module):
        def __init__(self, original, start, length, axis=-1):
            super().__init__()
            self.original = original
            self.start, self.length, self.axis = start, length, axis

        def forward(self, *args, **kwargs):
            output = self.original(*args, **kwargs)
            if isinstance(output, tuple):
                return (output[0].narrow(self.axis, self.start, self.length), *output[1:])
            return output.narrow(self.axis, self.start, self.length)

    view = copy.copy(branch)
    view._modules = branch._modules.copy()
    view._parameters = branch._parameters.copy()
    view._buffers = branch._buffers.copy()
    view.beta_proj = Select(branch.beta_proj, first, count)
    view.alpha = Select(branch.alpha, first, count, axis=1)
    view.output_gate = Select(branch.output_gate, first * head_dim, count * head_dim)
    if branch.short_conv is not None:
        convolution = copy.copy(branch.short_conv)
        convolution._modules = branch.short_conv._modules.copy()
        for name in branch.short_conv.targets:
            for suffix in ('sp', 'tm'):
                original = getattr(branch.short_conv, f'{name}_{suffix}')
                leaf = copy.copy(original)
                leaf._parameters = original._parameters.copy()
                leaf.weight = torch.nn.Parameter(original.weight.narrow(
                    0, first * head_dim, count * head_dim), requires_grad=False)
                assert leaf.weight.untyped_storage().data_ptr() == original.weight.untyped_storage().data_ptr()
                setattr(convolution, f'{name}_{suffix}', leaf)
        view.short_conv = convolution
    return view


def sequence_linear(self, out, x, q, k, v, window, *, group, rank, world):
    """Restore full sequence/head shard, call native branch, restore local queries."""
    import torch
    import torch.distributed as dist

    from h3_ascend.integrations.sequence_communication import exchange_output, exchange_qkv

    assert x.ndim == 2 and q.shape[0] == x.shape[0] and q.shape == k.shape == v.shape
    local_tokens, total_heads, width = q.shape
    assert total_heads % world == 0
    local_heads = total_heads // world
    full_x = torch.empty((local_tokens * world, x.shape[-1]), dtype=x.dtype, device=x.device)
    # Start the input gather before raw-QKV exchanges, then wait only when
    # frame statistics need x. All ranks issue identical collective order.
    gather = dist.all_gather_into_tensor(full_x, x.contiguous(), group=group, async_op=True)
    raw = exchange_qkv(tuple(t.unsqueeze(0) for t in (q, k, v)), group, world, mode='auto')
    gather.wait()
    key = (rank, world, width)
    if os.getenv('H3_ASCEND_VDN_SP_CACHE_VIEWS', '1') == '1':
        cache = self.__dict__.setdefault('_h3_sequence_views', {})
        if key not in cache:
            cache[key] = branch_head_view(self.linear_attention, rank * local_heads, local_heads, width)
        branch = cache[key]
    else:
        # Offloaded modules can replace storage on each entry. Do not retain
        # a view of an old staging buffer across layer-offload boundaries.
        branch = branch_head_view(self.linear_attention, rank * local_heads, local_heads, width)
    result = branch(full_x, *(t[0] for t in raw), window)
    self._h3_vdn_sequence_calls = getattr(self, '_h3_vdn_sequence_calls', 0) + 1
    if result is None:
        return out
    rows, readout = result
    all_rows = torch.zeros((1, local_tokens * world, local_heads, width), dtype=q.dtype, device=q.device)
    all_rows[0, rows] = readout.reshape(-1, local_heads, width)
    del full_x, raw, result, readout
    local_readout = exchange_output(all_rows, group, world)[0].reshape(local_tokens, total_heads * width)
    # Preserve zero contribution to text/audio/anchor rows. Do not quantize
    # padded zeros as though they were part of the original video projection.
    shard_start = rank * local_tokens
    lo = max(rows.start, shard_start)
    hi = min(rows.stop, shard_start + local_tokens)
    if lo < hi:
        selected = slice(lo - shard_start, hi - shard_start)
        out[selected] += self.to_out_linear(local_readout[selected])[0]
    return out


def install():
    if os.getenv('H3_ASCEND_VDN_SP') != '1':
        return
    import torch
    from vllm_omni.diffusion.models.minimax_h3 import vdnh3 as model

    if getattr(model.VDNH3HybridAttention, '_h3_sequence_adapter', False):
        return
    # Long-temporal native VDN repeats vary without deterministic NPU
    # execution. The eight-rank real-weight qualification uses this same
    # setting; no inverse, feature, scan or attention kernel is replaced.
    torch.use_deterministic_algorithms(True)
    original_check = model.VDNCheckpoint.check_serving_contract
    original_linear = model.VDNH3HybridAttention.add_linear

    def check(self, *, partition, od_config):
        parallel = od_config.parallel_config
        if int(parallel.ulysses_degree) <= 1:
            return original_check(self, partition=partition, od_config=od_config)
        if any(int(getattr(parallel, name, 1) or 1) != 1 for name in ('ring_degree', 'allgather_degree', 'tensor_parallel_size')):
            raise ValueError('VDN sequence adapter requires pure Ulysses, TP=1')
        checked = copy.copy(od_config)
        checked.parallel_config = copy.copy(parallel)
        checked.parallel_config.ulysses_degree = 1
        # Keep every original checkpoint/backend/offload check. Only replace
        # the unsupported-SP guard with this explicitly installed adapter.
        return original_check(self, partition=partition, od_config=checked)

    @torch.compiler.disable
    def add_linear(self, out, x, q, k, v, window):
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_sequence_parallel_world_size,
            get_sp_group,
        )

        if get_sequence_parallel_world_size() == 1:
            return original_linear(self, out, x, q, k, v, window)
        group = get_sp_group()
        if group.ring_world_size != 1 or group.allgather_world_size != 1:
            raise ValueError('VDN sequence adapter only supports pure Ulysses')
        return sequence_linear(self, out, x, q, k, v, window,
                               group=group.ulysses_group, rank=group.ulysses_rank,
                               world=group.ulysses_world_size)

    model.VDNCheckpoint.check_serving_contract = check
    model.VDNH3HybridAttention.add_linear = add_linear
    model.VDNH3HybridAttention._h3_sequence_adapter = True
