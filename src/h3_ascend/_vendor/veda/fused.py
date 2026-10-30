
"""Ascend fused gather, BF16 padding and FP32 mean/max/min; no changed math."""
import importlib as _importlib
import torch
import triton
import triton.language as tl

@triton.jit
def _pack(Q, K, V, G, S, HEADS, QT, KT, VT, FQ, FK, QS: tl.constexpr, QH: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr, N: tl.constexpr, D: tl.constexpr, C: tl.constexpr):
    tile = tl.program_id(0)
    lh = tl.program_id(1)
    c = tl.program_id(2) // (128 // C)
    head = tl.load(HEADS + lh)
    r = tl.arange(0, 128)
    d = tl.program_id(2) % (128 // C) * C + tl.arange(0, C)
    count = tl.load(S + tile)
    src = tl.load(G + tile * 128 + r)
    valid = r[:, None] < count
    if c == 0:
        x = tl.load(Q + src[:, None] * QS + head * QH + d[None, :], valid, 0)
    elif c == 1:
        x = tl.load(K + src[:, None] * KS + head * KH + d[None, :], valid, 0)
    else:
        x = tl.load(V + src[:, None] * VS + head * VH + d[None, :], valid, 0)
    pos = (lh * N * 128 + tile * 128 + r[:, None]) * D + d[None, :]
    if c == 0:
        tl.store(QT + pos, x)
    elif c == 1:
        tl.store(KT + pos, x)
    else:
        tl.store(VT + pos, x)
    if c < 2:
        xf = x.to(tl.float32)
        mean = tl.sum(xf, axis=0) / tl.maximum(count, 1)
        mx = tl.max(tl.where(valid, xf, float('-inf')), axis=0)
        mn = tl.min(tl.where(valid, xf, float('inf')), axis=0)
        fp = (lh * N + tile) * 3 * D + d
        if c == 0:
            tl.store(FQ + fp, tl.where(count > 0, mean, 0))
            tl.store(FQ + fp + D, tl.where(count > 0, mx, 0))
            tl.store(FQ + fp + 2 * D, tl.where(count > 0, mn, 0))
        else:
            tl.store(FK + fp, tl.where(count > 0, mean, 0))
            tl.store(FK + fp + D, tl.where(count > 0, mx, 0))
            tl.store(FK + fp + 2 * D, tl.where(count > 0, mn, 0))

def pack(q, k, v, layout, heads, channel_block=32):
    n = layout.n_tiles
    h = heads.numel()
    d = q.shape[-1]
    tiles = [torch.empty((1, h, n * 128, d), dtype=q.dtype, device=q.device) for _ in range(3)]
    feats = [torch.empty((h, n, 3 * d), dtype=torch.float32, device=q.device) for _ in range(2)]
    _pack[n, h, 3 * (128 // channel_block)](q, k, v, layout.gather_index, layout.valid_count, heads, *tiles, *feats, *q.stride()[:2], *k.stride()[:2], *v.stride()[:2], n, d, channel_block)
    return (*tiles, *feats)

@triton.jit
def _scatter(IN, INV, HEADS, OUT, S: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    lh = tl.program_id(1)
    d = tl.arange(0, D)
    head = tl.load(HEADS + lh)
    src = tl.load(INV + row, row < S, 0)
    x = tl.load(IN + (lh * N * 128 + src[:, None]) * D + d[None, :], row[:, None] < S, 0)
    tl.store(OUT + (row[:, None] * H + head) * D + d[None, :], x, row[:, None] < S)

def scatter(x, inv, heads, out, rows=32):
    _scatter[triton.cdiv(inv.numel(), rows), heads.numel()](x, inv, heads, out, inv.numel(), x.shape[2] // 128, out.shape[1], out.shape[2], rows)

@triton.jit
def _transport(M, O, N: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    source = i // (2 * N) * N + i % (2 * N) // 2
    x = tl.load(M + source, i < TOTAL, 0).to(tl.int8)
    tl.store(O + i, x, i < TOTAL)

def transport(mask):
    assert mask.is_contiguous()
    h, r, n = mask.shape
    out = torch.empty((1, h, r, 2 * n), device=mask.device, dtype=torch.int8)
    _transport[triton.cdiv(out.numel(), 4096),](mask, out, n, out.numel(), 4096)
    return out

@triton.jit
def _transport_rows(M, O, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(M + row * N + c // 2, c < 2 * N, 0).to(tl.int8)
    tl.store(O + row * 2 * N + c, x, c < 2 * N)

def transport_rows(mask):
    assert mask.is_contiguous()
    h, r, n = mask.shape
    out = torch.empty((1, h, r, 2 * n), device=mask.device, dtype=torch.int8)
    _transport_rows[h * r,](mask, out, n, triton.next_power_of_2(2 * n))
    return out
