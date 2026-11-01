"""Reuse head-local full-sequence raw QKV in the unchanged official VDN branch."""

import os


def sequence_linear_preprojected(self, out, x, q, k, v, window, *, group, rank, world):
    import torch
    import torch.distributed as dist
    from h3_ascend.integrations.sequence_communication import exchange_output
    from h3_ascend.integrations.vdn_sequence import branch_head_view

    if q.shape != k.shape or q.shape != v.shape or q.shape[0] != world * x.shape[0]:
        raise ValueError("Expected full-sequence head-local raw QKV")
    total, heads, width = q.shape
    if heads * world != 56 or width != 128:
        raise ValueError("Only original 56-head,128D VDN is qualified")
    full_x = torch.empty((total, x.shape[1]), dtype=x.dtype, device=x.device)
    # Beta/alpha/gates keep the exact original BF16 input and original projections.
    dist.all_gather_into_tensor(full_x, x.contiguous(), group=group)
    key = (rank, world, width)
    if os.getenv("H3_ASCEND_VDN_SP_CACHE_VIEWS", "1") == "1":
        cache = self.__dict__.setdefault("_h3_sequence_views", {})
        if key not in cache:
            cache[key] = branch_head_view(self.linear_attention, rank * heads, heads, width)
        branch = cache[key]
    else:
        branch = branch_head_view(self.linear_attention, rank * heads, heads, width)
    result = branch(full_x, q.contiguous(), k.contiguous(), v.contiguous(), window)
    self._h3_vdn_sequence_calls = getattr(self, "_h3_vdn_sequence_calls", 0) + 1
    self._h3_vdn_preprojected_calls = getattr(self, "_h3_vdn_preprojected_calls", 0) + 1
    if result is None:
        return out
    rows, readout = result
    global_readout = torch.zeros((1, total, heads, width), dtype=q.dtype, device=q.device)
    global_readout[0, rows] = readout.reshape(-1, heads, width)
    local_readout = exchange_output(global_readout, group, world)[0].reshape(x.shape[0], 56 * width)
    local_start = rank * x.shape[0]
    lo, hi = max(rows.start, local_start), min(rows.stop, local_start + x.shape[0])
    if lo < hi:
        selected = slice(lo - local_start, hi - local_start)
        out[selected] += self.to_out_linear(local_readout[selected])[0]
    return out


def install():
    from vllm_omni.diffusion.models.minimax_h3.vdnh3 import VDNH3HybridAttention
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

    if getattr(VDNH3HybridAttention, "_h3_input_qkv_adapter", False):
        return
    original = VDNH3HybridAttention.add_linear

    def add(self, out, x, q, k, v, window):
        group = get_sp_group()
        world = group.ulysses_world_size
        if world > 1 and q.shape[0] == x.shape[0] * world and q.shape[1] * world == 56:
            return sequence_linear_preprojected(
                self,
                out,
                x,
                q,
                k,
                v,
                window,
                group=group.ulysses_group,
                rank=group.ulysses_rank,
                world=world,
            )
        return original(self, out, x, q, k, v, window)

    VDNH3HybridAttention.add_linear = add
    VDNH3HybridAttention._h3_input_qkv_adapter = True
