
"""Spark's official BF16 K-centroid reduction and exact low-ID Top-K ties.
The centroid order mirrors upstream SM120's PTX: eight sequential row-register
adds with BF16 rounding, then row-bit reductions 0, 2, 1 (4-warp config).
"""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl

@triton.jit
def _rows(a, r):
    return tl.gather(a, tl.broadcast_to(r[:, None], (r.shape[0], 128)), axis=0)

@triton.jit
def _bf16_sum8(a):
    a4 = (_rows(a, tl.arange(0, 4) * 2).to(tl.float32) + _rows(a, tl.arange(0, 4) * 2 + 1).to(tl.float32)).to(tl.bfloat16)
    a2 = (_rows(a4, tl.arange(0, 2)).to(tl.float32) + _rows(a4, tl.arange(0, 2) + 2).to(tl.float32)).to(tl.bfloat16)
    a1 = (_rows(a2, tl.arange(0, 1)).to(tl.float32) + _rows(a2, tl.arange(0, 1) + 1).to(tl.float32)).to(tl.bfloat16)
    return a1.reshape(128)

@triton.jit
def _kc(K, O, N: tl.constexpr, M: tl.constexpr):
    h = tl.program_id(1)
    kb = tl.program_id(0)
    r = kb * 64 + tl.arange(0, 8)
    d = tl.arange(0, 128)
    a = tl.load(K + (h * N + r[:, None]) * 128 + d[None, :])
    for j in tl.static_range(1, 8):
        b = tl.load(K + (h * N + r[:, None] + j * 8) * 128 + d[None, :])
        a = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    mean = (_bf16_sum8(a).to(tl.float32) / 64.0).to(tl.bfloat16)
    tl.store(O + (h * M + kb) * 128 + d, mean)

def centroids(k, m):
    h, n, d = k.shape
    out = torch.empty((h, m, 128), device=k.device, dtype=torch.bfloat16)
    _kc[m, h](k, out, n, m)
    return out

@triton.jit
def _mask(S, VALUES, R, M: tl.constexpr, W: tl.constexpr, P: tl.constexpr, SORT: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, P)
    v = tl.load(S + row * M + j, j < M, -float('inf')).to(tl.float32)
    if SORT:
        ordered = tl.sort(v, descending=True)
        cut = tl.sum(tl.where(j == W - 1, ordered, 0.0), 0)
    else:
        cut = tl.load(VALUES + row * W + W - 1)
    greater = (j < M) & (v > cut)
    eq = (j < M) & (v == cut)
    need = W - tl.sum(greater.to(tl.int32), 0)
    rank = tl.cumsum(eq.to(tl.int32), 0)
    keep = greater | eq & (rank <= need)
    tl.store(R + row * M + j, keep.to(tl.int8), j < M)

def route(scores, w, sort=False):
    h, m, _ = scores.shape
    r = torch.empty((h, m, m), device=scores.device, dtype=torch.int8)
    if sort:
        vals = r
        ids = torch.empty(1, device=scores.device, dtype=torch.int32).expand(h, m, w)
    else:
        vals, ids = scores.topk(w, sorted=True)
        ids = ids.int()
    _mask[h * m,](scores, vals, r, m, w, triton.next_power_of_2(m), sort)
    return (ids, r)

@triton.jit
def _kc_split(K, O, N: tl.constexpr, M: tl.constexpr):
    h = tl.program_id(1)
    kb = tl.program_id(0)
    d = tl.arange(0, 128)
    ptr = tl.make_block_ptr(K + h * N * 128, (N, 128), (128, 1), (kb * 64, 0), (64, 128), (1, 0))
    x = tl.load(ptr)
    z = tl.trans(x.reshape(8, 8, 128), (1, 2, 0)).reshape(8, 128, 4, 2)
    e, o = tl.split(z)
    e04, e26 = tl.split(e.reshape(8, 128, 2, 2))
    o15, o37 = tl.split(o.reshape(8, 128, 2, 2))
    r0, r4 = tl.split(e04)
    r2, r6 = tl.split(e26)
    r1, r5 = tl.split(o15)
    r3, r7 = tl.split(o37)
    a = (r0.to(tl.float32) + r1.to(tl.float32)).to(tl.bfloat16)
    a = (a.to(tl.float32) + r2.to(tl.float32)).to(tl.bfloat16)
    a = (a.to(tl.float32) + r3.to(tl.float32)).to(tl.bfloat16)
    a = (a.to(tl.float32) + r4.to(tl.float32)).to(tl.bfloat16)
    a = (a.to(tl.float32) + r5.to(tl.float32)).to(tl.bfloat16)
    a = (a.to(tl.float32) + r6.to(tl.float32)).to(tl.bfloat16)
    a = (a.to(tl.float32) + r7.to(tl.float32)).to(tl.bfloat16)
    even, odd = tl.split(tl.trans(a.reshape(4, 2, 128), (0, 2, 1)))
    a4 = (even.to(tl.float32) + odd.to(tl.float32)).to(tl.bfloat16)
    lo, hi = tl.split(tl.trans(a4.reshape(2, 2, 128), (1, 2, 0)))
    a2 = (lo.to(tl.float32) + hi.to(tl.float32)).to(tl.bfloat16)
    lo, hi = tl.split(a2.T)
    total = (lo.to(tl.float32) + hi.to(tl.float32)).to(tl.bfloat16)
    tl.store(O + (h * M + kb) * 128 + d, (total.to(tl.float32) / 64.0).to(tl.bfloat16))

def centroids_split(k, m):
    h, n, d = k.shape
    out = torch.empty((h, m, 128), device=k.device, dtype=torch.bfloat16)
    _kc_split[m, h](k, out, n, m)
    return out
