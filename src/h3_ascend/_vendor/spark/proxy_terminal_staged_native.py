
"""Exact terminal proxy: short vector stages separated by native stable Sort32."""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
from . import select32_native
import triton.language as tl
from .proxy_hybrid_ranklayout import _unit

@triton.jit
def _seed(C, D, U, V, B: tl.constexpr):
    p = tl.program_id(0) * 2 + tl.arange(0, 2)
    i = tl.arange(0, 32)
    d = tl.arange(0, 128)
    center = tl.load(C + (p[:, None, None] * 32 + i[None, :, None]) * 128 + d[None, None, :], (p < B)[:, None, None], 0).to(tl.float32)
    unit = center / tl.maximum(tl.sqrt(tl.sum(center * center, 2)), 1e-12)[:, :, None]
    dist = tl.load(D + (p[:, None, None] * 32 + i[None, :, None]) * 32 + i[None, None, :], (p < B)[:, None, None], 0)
    masked = tl.where((i[:, None] < i[None, :])[None, :, :], dist, -float('inf'))
    maximum = tl.max(tl.max(masked, 2), 1)
    pi = i[:, None] * 32 + i[None, :]
    candidates = tl.where(masked == maximum[:, None, None], pi[None, :, :], 1024)
    farthest = tl.min(tl.min(candidates, 2), 1)
    left = tl.sum(tl.where(i[None, :, None] == (farthest // 32)[:, None, None], center, 0.0), 1)
    right = tl.sum(tl.where(i[None, :, None] == (farthest % 32)[:, None, None], center, 0.0), 1)
    direction = _unit(right) - _unit(left)
    delta = tl.sum(unit * direction[:, None, :], 2)
    tl.store(U + (p[:, None, None] * 32 + i[None, :, None]) * 128 + d[None, None, :], unit, (p < B)[:, None, None])
    tl.store(V + p[:, None] * 32 + i[None, :], delta, (p < B)[:, None])

@triton.jit
def _update(C, U, I, O, B: tl.constexpr, FINAL: tl.constexpr):
    p = tl.program_id(0) * 2 + tl.arange(0, 2)
    i = tl.arange(0, 32)
    j = tl.arange(0, 16)
    d = tl.arange(0, 128)
    chosen = tl.load(I + p[:, None] * 32 + j[None, :], (p < B)[:, None], 0)
    selected = tl.sum((i[None, :, None] == chosen[:, None, :]).to(tl.int32), 2) > 0
    lw = tl.where(selected, 4, 0)
    rw = 4 - lw
    center = tl.load(C + (p[:, None, None] * 32 + i[None, :, None]) * 128 + d[None, None, :], (p < B)[:, None, None], 0).to(tl.float32)
    left = tl.sum(center * lw[:, :, None].to(tl.float32), 1) / 64
    right = tl.sum(center * rw[:, :, None].to(tl.float32), 1) / 64
    direction = _unit(right) - _unit(left)
    if FINAL:
        tl.store(O + p[:, None] * 128 + d[None, :], direction, (p < B)[:, None])
    else:
        unit = tl.load(U + (p[:, None, None] * 32 + i[None, :, None]) * 128 + d[None, None, :], (p < B)[:, None, None], 0)
        delta = tl.sum(unit * direction[:, None, :], 2)
        tl.store(O + p[:, None] * 32 + i[None, :], delta, (p < B)[:, None])

def directions(c, weights, caps, bp=2):
    assert caps == (64, 64) and bp == 2
    b = c.shape[0]
    u = c.float()
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dist = (1 - torch.bmm(u, u.transpose(1, 2))).contiguous()
    unit = torch.empty_like(c, dtype=torch.float32)
    delta = torch.empty((b, 32), device=c.device, dtype=torch.float32)
    delta2 = torch.empty_like(delta)
    out = torch.empty((b, 1, 128), device=c.device, dtype=torch.float32)
    _seed[triton.cdiv(b, 2),](c, dist, unit, delta, b)
    order = select32_native.run(delta)
    _update[triton.cdiv(b, 2),](c, unit, order, delta2, b, False)
    order2 = select32_native.run(delta2)
    _update[triton.cdiv(b, 2),](c, unit, order2, out, b, True)
    return out
