
"""Batch independent proxy-tree parents in each vector program; same FP32 math."""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import os
import triton.language as tl
from . import proxy_fused
from .proxy_fused import _unit_row

@triton.jit
def _unit(v):
    d = tl.arange(0, 128)
    a = tl.max(tl.where((tl.arange(0, 2) == 0)[:, None], v, -float('inf')), 0)
    b = tl.max(tl.where((tl.arange(0, 2) == 1)[:, None], v, -float('inf')), 0)
    ua = _unit_row(a)
    ub = _unit_row(b)
    return tl.where((tl.arange(0, 2) == 0)[:, None], ua[None, :], ub[None, :])

@triton.jit
def _split(center, unit, dist, weight, lc, rc, L: tl.constexpr):
    i = tl.arange(0, L)
    pv = (i[:, None] < i[None, :])[None, :, :] & (weight[:, :, None] > 0) & (weight[:, None, :] > 0)
    masked = tl.where(pv, dist, -float('inf'))
    maximum = tl.max(tl.max(masked, 2), 1)
    pi = i[:, None] * L + i[None, :]
    candidates = tl.where(masked == maximum[:, None, None], pi[None, :, :], L * L)
    farthest = tl.min(tl.min(candidates, 2), 1)
    active = tl.sum((weight > 0).to(tl.int32), 1)
    only = tl.argmax((weight > 0).to(tl.int32), 1, tie_break_left=True)
    first = tl.where(active < 2, only, farthest // L)
    second = tl.where(active < 2, only, farthest % L)
    left = tl.sum(tl.where(i[None, :, None] == first[:, None, None], center, 0.0), 1)
    right = tl.sum(tl.where(i[None, :, None] == second[:, None, None], center, 0.0), 1)
    lw = weight
    rw = weight
    for _ in tl.static_range(2):
        direction = _unit(right) - _unit(left)
        delta = tl.sum(unit * direction[:, None, :], 2)
        ordered = tl.sort(delta, descending=False, dim=1)
        cut = tl.sum(tl.where(i[None, :] == 15, ordered, 0.0), 1)
        below = delta < cut[:, None]
        equal = delta == cut[:, None]
        need = 16 - tl.sum(below.to(tl.int32), 1)
        selected = below | equal & (tl.cumsum(equal.to(tl.int32), 1) <= need[:, None])
        lw = tl.where(selected, 4, 0).to(tl.int32)
        rw = 4 - lw
        left = tl.sum(center * lw[:, :, None].to(tl.float32), 1) / lc
        right = tl.sum(center * rw[:, :, None].to(tl.float32), 1) / rc
    return (_unit(right) - _unit(left), lw, rw)

@triton.jit
def _kernel(C, W, D, LC, RC, O, B: tl.constexpr, L: tl.constexpr, INTERNAL: tl.constexpr, SLOTS: tl.constexpr, BP: tl.constexpr):
    parents = tl.program_id(0) * BP + tl.arange(0, BP)
    i = tl.arange(0, L)
    d = tl.arange(0, 128)
    center = tl.load(C + (parents[:, None, None] * L + i[None, :, None]) * 128 + d[None, None, :], (parents < B)[:, None, None], 0).to(tl.float32)
    unit = center / tl.maximum(tl.sqrt(tl.sum(center * center, 2)), 1e-12)[:, :, None]
    dist = tl.load(D + (parents[:, None, None] * L + i[None, :, None]) * L + i[None, None, :], (parents < B)[:, None, None], 0)
    root = tl.full((BP, L), 4, tl.int32)
    direction, lw, rw = _split(center, unit, dist, root, 64, 64, L)
    tl.store(O + parents[:, None] * INTERNAL * 128 + d[None, :], direction, (parents < B)[:, None])

def directions(centers, weights, caps, bp=2):
    assert bp == 2 and caps == (64, 64)
    if _settings.get('PROXY_DIST') == 'raw_bf16':
        dist = _importlib.import_module('h3_ascend._vendor.spark.proxy_dist').raw(centers)
    else:
        u = centers.float()
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        dist = (1 - torch.bmm(u, u.transpose(1, 2))).contiguous()
    top = proxy_fused.topology(caps)
    out = torch.empty((centers.shape[0], len(top), 128), device=centers.device, dtype=torch.float32)
    key = (tuple(caps), str(centers.device))
    if key not in proxy_fused._CAP_CACHE:
        proxy_fused._CAP_CACHE[key] = [torch.tensor([t[i] for t in top], device=centers.device, dtype=torch.int32) for i in (1, 2)]
    lc, rc = proxy_fused._CAP_CACHE[key]
    _kernel[triton.cdiv(centers.shape[0], bp),](centers, weights[0].contiguous().int(), dist, lc, rc, out, centers.shape[0], 32, len(top), triton.next_power_of_2(2 * len(top) + 1), bp)
    return out
