
"""Exact native pooling/top-k, task-local Ascend C KV64 or paired KV128."""
import importlib as _importlib
import os
import math
import torch
import torch_npu
import triton
import triton.language as tl
from .tile_geometry import geometry
from .sparse64_ascendc import load_library

@triton.jit
def _finish(O, C, G, INV, R, REAL: tl.constexpr, H: tl.constexpr, HH: tl.constexpr, D: tl.constexpr, NT: tl.constexpr, GS: tl.constexpr, GH: tl.constexpr, BNSD: tl.constexpr):
    row = tl.program_id(0)
    h = tl.arange(0, HH)
    d = tl.arange(0, D)
    token = tl.load(INV + row, row < REAL, 0)
    valid = (row < REAL) & (h[:, None] < H)
    if BNSD:
        offset = h[:, None] * NT * 64 * D + token * D + d[None, :]
    else:
        offset = token * H * D + h[:, None] * D + d[None, :]
    raw = tl.load(O + offset, valid, 0).to(tl.float32)
    compressed = tl.load(C + h[:, None] * NT * D + token // 64 * D + d[None, :], valid, 0).to(tl.float32)
    gate = tl.load(G + row * GS + h[:, None] * GH + d[None, :], valid, 0).to(tl.float32)
    result = raw + (compressed * gate).to(tl.bfloat16).to(tl.float32)
    tl.store(R + row * H * D + h[:, None] * D + d[None, :], tl.where(row < REAL, result, 0.0), h[:, None] < H)

def forward(q, k, v, gate, prefix_segments, grid, group=2, prefix_fa=False, bm=64, options=None, keep_rate=0.1, topk=None):
    load_library()
    options = dict(options or {})
    stack = options.pop('route_stack_tiles', 1)
    layout = options.pop('route_layout', 'BNSD')
    joint = options.pop('route_joint', False)
    assert not options, options
    sizes, gather, valid, inv, nprefix = geometry(tuple(prefix_segments), tuple(grid), q.device, (4, 4, 4))
    nt, h, d = (sizes.numel(), q.shape[1], q.shape[2])

    def tile(x):
        return (x.index_select(0, gather).view(nt, 64, h, d) * valid[:, :, None, None]).reshape(nt * 64, h, d)
    if joint:
        from .vsa64_ascendc_tile import tile_pool
        qt, kt, vt, qp, kp, vp = tile_pool(q, k, v, gather, sizes)
    else:
        qt, kt, vt = [tile(x) for x in (q, k, v)]

        def pool(x):
            return (x.view(nt, 64, h, d).sum(1, dtype=torch.float32) / sizes[:, None, None]).permute(1, 0, 2)
        qp, kp, vp = [pool(x) for x in (qt, kt, vt)]
    scores = qp @ kp.transpose(-2, -1) * d ** (-0.5)
    start = nprefix if prefix_fa else 0
    if not 0 < keep_rate <= 1:
        raise ValueError('VSA keep rate must lie in (0, 1]')
    keep = min(nt - nprefix, max(1, int(topk) if topk is not None else math.ceil(keep_rate * (nt - nprefix))))
    chosen = scores[:, start:, nprefix:].topk(keep, -1).indices + nprefix
    mask = torch.zeros((1, h, nt - start, nt), device=q.device, dtype=torch.int8)
    mask.scatter_(-1, chosen.unsqueeze(0), 1)
    mask[:, :, :, :nprefix] = 1
    if not prefix_fa:
        mask[:, :, :nprefix, :] = 1
    if layout == 'BNSD':
        qb, kb, vb = [x.transpose(0, 1).unsqueeze(0).contiguous() for x in (qt, kt, vt)]
        query = qb[:, :, start * 64:].contiguous()
        raw, lse, _ = torch.ops.sol_h3_bsa_ext.forward(query, kb, vb, mask, sizes, [64, 64], None, None, layout, h, d ** (-0.5), 0, True, group * 64, stack)
        if prefix_fa:
            kr, vr = [x[:inv.numel()].transpose(0, 1).unsqueeze(0).contiguous() for x in (k, v)]
            dense = torch_npu.npu_fusion_attention(qb[:, :, :nprefix * 64].contiguous(), kr, vr, h, 'BNSD', scale=d ** (-0.5))[0]
            raw = torch.cat((dense, raw), 2)
    else:
        query = qt[start * 64:].clone() if start else qt
        raw, lse, _ = torch.ops.sol_h3_bsa_ext.forward(query, kt, vt, mask, sizes, [64, 64], [query.shape[0]], [kt.shape[0]], layout, h, d ** (-0.5), 0, True, group * 64, stack)
        if prefix_fa:
            qp = qt[:nprefix * 64].transpose(0, 1).unsqueeze(0).contiguous()
            kr, vr = [x[:inv.numel()].transpose(0, 1).unsqueeze(0).contiguous() for x in (k, v)]
            dense = torch_npu.npu_fusion_attention(qp, kr, vr, h, 'BNSD', scale=d ** (-0.5))[0].squeeze(0).transpose(0, 1)
            raw = torch.cat((dense, raw), 0)
    compressed = (scores.softmax(-1) @ vp).to(q.dtype)
    result = torch.empty_like(q)
    _finish[q.shape[0],](raw, compressed, gate, inv, result, inv.numel(), h, triton.next_power_of_2(h), d, nt, *gate.stride()[:2], layout == 'BNSD', enable_fp_fusion=False)
    return result
