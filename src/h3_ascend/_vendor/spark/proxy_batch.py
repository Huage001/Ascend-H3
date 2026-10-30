
"""Batch independent proxy-tree parents in each vector program; same FP32 math."""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import os
import triton.language as tl
from . import proxy_fused

@triton.jit
def _unit(v):
    return v / tl.maximum(tl.sqrt(tl.sum(v * v, 1)), 1e-12)[:, None]

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
        before = (delta[:, None, :] < delta[:, :, None]) | (delta[:, None, :] == delta[:, :, None]) & (i[None, None, :] < i[None, :, None])
        prefix = tl.sum(tl.where(before, weight[:, None, :], 0), 2)
        lw = tl.minimum(tl.maximum(lc - prefix, 0), weight)
        rw = weight - lw
        left = tl.sum(center * lw[:, :, None].to(tl.float32), 1) / lc
        right = tl.sum(center * rw[:, :, None].to(tl.float32), 1) / rc
    return (_unit(right) - _unit(left), lw, rw)

@triton.jit
def _kernel(C, W, D, LC, RC, O, B: tl.constexpr, L: tl.constexpr, INTERNAL: tl.constexpr, SLOTS: tl.constexpr, BP: tl.constexpr):
    parents = tl.program_id(0) * BP + tl.arange(0, BP)
    i = tl.arange(0, L)
    d = tl.arange(0, 128)
    slots = tl.arange(0, SLOTS)
    center = tl.load(C + (parents[:, None, None] * L + i[None, :, None]) * 128 + d[None, None, :], (parents < B)[:, None, None], 0).to(tl.float32)
    unit = center / tl.maximum(tl.sqrt(tl.sum(center * center, 2)), 1e-12)[:, :, None]
    dist = tl.load(D + (parents[:, None, None] * L + i[None, :, None]) * L + i[None, None, :], (parents < B)[:, None, None], 0)
    root = tl.load(W + i).to(tl.int32)
    nodes = tl.where((slots == 0)[None, :, None], root[None, None, :], 0)
    nodes = tl.broadcast_to(nodes, (BP, SLOTS, L))
    for node in range(INTERNAL):
        weight = tl.sum(tl.where((slots == node)[None, :, None], nodes, 0), 1)
        direction, lw, rw = _split(center, unit, dist, weight, tl.load(LC + node), tl.load(RC + node), L)
        nodes = tl.where((slots == 2 * node + 1)[None, :, None], lw[:, None, :], nodes)
        nodes = tl.where((slots == 2 * node + 2)[None, :, None], rw[:, None, :], nodes)
        tl.store(O + (parents[:, None] * INTERNAL + node) * 128 + d[None, :], direction, (parents < B)[:, None])

def directions(centers, weights, caps, bp=2):
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
