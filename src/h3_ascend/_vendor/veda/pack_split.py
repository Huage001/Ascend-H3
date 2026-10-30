
"""Larger BF16 gather plus contiguous FP32 pool, compared with fused gather-pool."""
import importlib as _importlib
import torch
import triton
import triton.language as tl

@triton.jit
def _gather(Q, K, V, G, S, H, QT, KT, VT, QS: tl.constexpr, QH: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr, N: tl.constexpr):
    tile = tl.program_id(0)
    lh = tl.program_id(1)
    kind = tl.program_id(2)
    r = tl.arange(0, 128)
    d = tl.arange(0, 128)
    h = tl.load(H + lh)
    count = tl.load(S + tile)
    src = tl.load(G + tile * 128 + r)
    valid = r[:, None] < count
    if kind == 0:
        x = tl.load(Q + src[:, None] * QS + h * QH + d[None, :], valid, 0)
    elif kind == 1:
        x = tl.load(K + src[:, None] * KS + h * KH + d[None, :], valid, 0)
    else:
        x = tl.load(V + src[:, None] * VS + h * VH + d[None, :], valid, 0)
    dest = (lh * N * 128 + tile * 128 + r[:, None]) * 128 + d[None, :]
    if kind == 0:
        tl.store(QT + dest, x)
    elif kind == 1:
        tl.store(KT + dest, x)
    else:
        tl.store(VT + dest, x)

@triton.jit
def _pool(QT, KT, S, FQ, FK, N: tl.constexpr):
    tile = tl.program_id(0)
    lh = tl.program_id(1)
    kind = tl.program_id(2) // 2
    r = tl.arange(0, 128)
    d = tl.program_id(2) % 2 * 64 + tl.arange(0, 64)
    count = tl.load(S + tile)
    pos = (lh * N * 128 + tile * 128 + r[:, None]) * 128 + d[None, :]
    if kind == 0:
        x = tl.load(QT + pos).to(tl.float32)
    else:
        x = tl.load(KT + pos).to(tl.float32)
    valid = r[:, None] < count
    mean = tl.sum(x, 0) / tl.maximum(count, 1)
    mx = tl.max(tl.where(valid, x, float('-inf')), 0)
    mn = tl.min(tl.where(valid, x, float('inf')), 0)
    dest = (lh * N + tile) * 384 + d
    if kind == 0:
        tl.store(FQ + dest, tl.where(count > 0, mean, 0))
        tl.store(FQ + dest + 128, tl.where(count > 0, mx, 0))
        tl.store(FQ + dest + 256, tl.where(count > 0, mn, 0))
    else:
        tl.store(FK + dest, tl.where(count > 0, mean, 0))
        tl.store(FK + dest + 128, tl.where(count > 0, mx, 0))
        tl.store(FK + dest + 256, tl.where(count > 0, mn, 0))

def pack(q, k, v, layout, heads, channel_block=64):
    assert channel_block == 64
    n = layout.n_tiles
    h = heads.numel()
    tiles = [torch.empty((1, h, n * 128, 128), dtype=q.dtype, device=q.device) for _ in range(3)]
    feats = [torch.empty((h, n, 384), dtype=torch.float32, device=q.device) for _ in range(2)]
    _gather[n, h, 3](q, k, v, layout.gather_index, layout.valid_count, heads, *tiles, *q.stride()[:2], *k.stride()[:2], *v.stride()[:2], n)
    _pool[n, h, 4](tiles[0], tiles[1], layout.valid_count, *feats, n)
    return (*tiles, *feats)
