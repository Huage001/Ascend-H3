
"""Ascend-only exact-rule Spark primitives; BF16 model tensors, FP32 statistics."""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl

@triton.jit
def _chol_diag(A, C: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, 16)
    j = tl.arange(0, 16)
    ptr = A + b * 16384 + (C + i[:, None]) * 128 + C + j[None, :]
    a = tl.load(ptr)
    for c in range(16):
        column = tl.sum(tl.where(j[None, :] == c, a, 0.0), 1)
        diag = tl.sum(tl.where(i == c, column, 0.0), 0)
        col = column / tl.sqrt(diag)
        a = tl.where((i[:, None] > c) & (j[None, :] > c), a - col[:, None] * col[None, :], a)
        a = tl.where((j[None, :] == c) & (i[:, None] >= c), col[:, None], a)
    tl.store(ptr, a)

@triton.jit
def _chol_panel(A, C: tl.constexpr):
    b = tl.program_id(0)
    i = C + 16 + tl.arange(0, 128)
    j = tl.arange(0, 16)
    ptr = A + b * 16384 + i[:, None] * 128 + C + j[None, :]
    a = tl.load(ptr, i[:, None] < 128, 0)
    for c in range(16):
        lr = tl.load(A + b * 16384 + (C + j) * 128 + C + c)
        diag = tl.sum(tl.where(j == c, lr, 0.0), 0)
        col = tl.sum(tl.where(j[None, :] == c, a, 0.0), 1) / diag
        a = tl.where(j[None, :] > c, a - col[:, None] * lr[None, :], a)
        a = tl.where(j[None, :] == c, col[:, None], a)
    tl.store(ptr, a, i[:, None] < 128)

@triton.jit
def _chol_update(A, C: tl.constexpr):
    b = tl.program_id(2)
    r = C + 16 + tl.program_id(0) * 32 + tl.arange(0, 32)
    c = C + 16 + tl.program_id(1) * 32 + tl.arange(0, 32)
    d = tl.arange(0, 16)
    x = tl.load(A + b * 16384 + r[:, None] * 128 + C + d[None, :], r[:, None] < 128, 0)
    y = tl.load(A + b * 16384 + c[None, :] * 128 + C + d[:, None], c[None, :] < 128, 0)
    xh = x.to(tl.float16)
    yh = y.to(tl.float16)
    xl = ((x - xh.to(tl.float32)) * 4096.0).to(tl.float16)
    yl = ((y - yh.to(tl.float32)) * 4096.0).to(tl.float16)
    prod = tl.dot(xh, yh) + (tl.dot(xl, yh) + tl.dot(xh, yl)) * (1.0 / 4096.0)
    ptr = A + b * 16384 + r[:, None] * 128 + c[None, :]
    valid = (r[:, None] < 128) & (c[None, :] < 128) & (r[:, None] >= c[None, :])
    a = tl.load(ptr, valid, 0)
    tl.store(ptr, a - prod, valid)

def cholesky(a):
    out = a.clone()
    b = a.shape[0]
    for c in range(0, 128, 16):
        _chol_diag[b,](out, c)
        if c < 112:
            _chol_panel[b,](out, c)
            _chol_update[triton.cdiv(112 - c, 32), triton.cdiv(112 - c, 32), b](out, c)
    return out.tril()

@triton.jit
def _cosine_scores(X, I, D, O, N: tl.constexpr, S: tl.constexpr, BM: tl.constexpr, COLUMN_MAJOR: tl.constexpr=False):
    b = tl.program_id(1)
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, 128)
    c = tl.arange(0, 16)
    ids = tl.load(I + b * N + r, r < N, 0).to(tl.int64)
    x = tl.load(X + ids[:, None] * 128 + d[None, :], r[:, None] < N, 0).to(tl.float32)
    dirs = tl.load(D + (b * S + c[:, None]) * 128 + d[None, :], c[:, None] < S, 0)
    norm = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), 1e-12)
    score = tl.dot((x / norm[:, None]).to(tl.float16), dirs.T.to(tl.float16))
    if COLUMN_MAJOR:
        tl.store(O + (b * S + c[None, :]) * N + r[:, None], score, (r[:, None] < N) & (c[None, :] < S))
    else:
        tl.store(O + (b * N + r[:, None]) * S + c[None, :], score, (r[:, None] < N) & (c[None, :] < S))

def scores(x, ids, directions):
    column_major = _settings.get('SCORE_LAYOUT') == 'column'
    shape = (ids.shape[0], directions.shape[1], ids.shape[1]) if column_major else (*ids.shape, directions.shape[1])
    out = torch.empty(shape, device=x.device, dtype=torch.float32)
    bm = int(_settings.get('SCORE_BM', '64'))
    _cosine_scores[triton.cdiv(ids.shape[1], bm), ids.shape[0]](x, ids, directions, out, ids.shape[1], directions.shape[1], bm, column_major)
    return out.transpose(1, 2) if column_major else out

@triton.jit
def _permute(X, I, Y, N: tl.constexpr, G: tl.constexpr, H: tl.constexpr, T: tl.constexpr, INVERSE: tl.constexpr, B: tl.constexpr, P: tl.constexpr):
    b = tl.program_id(1)
    r = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, 128)
    idx = tl.load(I + b * N + tl.minimum(r, N - 1), r < N, 0)
    if not INVERSE:
        src = tl.where(r < N, G + idx, r - N)
        val = tl.load(X + (src[:, None] * H + b) * 128 + d[None, :], r[:, None] < T, 0)
        tl.store(Y + (b * P + r[:, None]) * 128 + d[None, :], val, r[:, None] < P)
    else:
        dest = tl.where(r < N, G + idx, r - N)
        val = tl.load(X + (b * P + r[:, None]) * 128 + d[None, :], r[:, None] < T, 0)
        tl.store(Y + (dest[:, None] * H + b) * 128 + d[None, :], val, r[:, None] < T)

def permute(x, ids, g, inverse=False, pad=0):
    h, n = ids.shape
    t = n + g
    p = x.shape[1] if inverse else triton.cdiv(t, pad) * pad if pad else t
    out = torch.empty((1, t, h, 128) if inverse else (h, p, 128), device=x.device, dtype=x.dtype)
    _permute[triton.cdiv(p, 64), h](x, ids, out, n, g, h, t, inverse, 64, p)
    return out

@triton.jit
def _summary(A, K, V, AK, AV, LM, N: tl.constexpr, M: tl.constexpr):
    block = tl.program_id(0)
    h = tl.program_id(1)
    r = block * 64 + tl.arange(0, 64)
    d = tl.arange(0, 128)
    p = tl.arange(0, 16)
    a = tl.load(A + h * 128 + d)
    k = tl.load(K + (h * N + r[:, None]) * 128 + d[None, :])
    v = tl.load(V + (h * N + r[:, None]) * 128 + d[None, :])
    am = tl.broadcast_to(a[None, :], (16, 128))
    score = tl.dot(am, k.T) * (0.08838834764831845 * 1.4426950408889634)
    mx = tl.max(score, 1)
    pr = tl.exp2(score - mx[:, None])
    den = tl.sum(pr, 1)
    ak = tl.dot(pr.to(tl.bfloat16), k) / den[:, None]
    av = tl.dot(pr.to(tl.bfloat16), v) / den[:, None]
    stored = ak.to(tl.bfloat16)
    lm = (mx + tl.log2(den)) * 0.6931471805599453 - tl.sum(am.to(tl.float32) * stored.to(tl.float32), 1) * 0.08838834764831845
    tl.store(AK + (h * M + block) * 128 + d, tl.sum(tl.where(p[:, None] == 0, stored.to(tl.float32), 0.0), 0).to(tl.bfloat16))
    tl.store(AV + (h * M + block) * 128 + d, tl.sum(tl.where(p[:, None] == 0, av, 0.0), 0).to(tl.bfloat16))
    tl.store(LM + h * M + block, tl.sum(tl.where(p == 0, lm, 0.0), 0))

def summaries(a, k, v, m):
    h, n, d = k.shape
    ak = torch.empty((h, m, 128), device=k.device, dtype=k.dtype)
    av = torch.empty_like(ak)
    lm = torch.empty((h, m), device=k.device, dtype=torch.float32)
    _summary[m, h](a, k, v, ak, av, lm, n, m)
    return (ak, av, lm)

@triton.jit
def _online(score, v, m, den, acc):
    nm = tl.maximum(m, tl.max(score, 1))
    alpha = tl.exp2(m - nm)
    p = tl.exp2(score - nm[:, None])
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    den = den * alpha + tl.sum(p, 1)
    return (nm, den, acc)

@triton.jit
def _spark(Q, K, V, I, R, AK, AV, LM, O, N: tl.constexpr, M: tl.constexpr, W: tl.constexpr, BM: tl.constexpr):
    qid = tl.program_id(0)
    h = tl.program_id(1)
    qb = qid // (64 // BM)
    ro = qid % (64 // BM) * BM
    rows = qb * 64 + ro + tl.arange(0, BM)
    d = tl.arange(0, 128)
    lane = tl.arange(0, 128)
    q = tl.load(Q + (h * N + rows[:, None]) * 128 + d[None, :])
    mx = tl.full((BM,), -float('inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 128), tl.float32)
    for s in range(M * 64, N, 128):
        r = s + lane
        k = tl.load(K + (h * N + r[None, :]) * 128 + d[:, None], r[None, :] < N, 0)
        v = tl.load(V + (h * N + r[:, None]) * 128 + d[None, :], r[:, None] < N, 0)
        score = tl.dot(q, k) * (0.08838834764831845 * 1.4426950408889634)
        score = tl.where((r < N)[None, :], score, -float('inf'))
        mx, den, acc = _online(score, v, mx, den, acc)
    for s in range(0, M, 128):
        r = s + lane
        ak = tl.load(AK + (h * M + r[None, :]) * 128 + d[:, None], r[None, :] < M, 0)
        av = tl.load(AV + (h * M + r[:, None]) * 128 + d[None, :], r[:, None] < M, 0)
        lm = tl.load(LM + h * M + r, r < M, 0)
        sel = tl.load(R + (h * M + qb) * M + r, r < M, 1)
        score = tl.dot(q, ak) * (0.08838834764831845 * 1.4426950408889634) + lm[None, :] * 1.4426950408889634
        score = tl.where(((r < M) & (sel == 0))[None, :], score, -float('inf'))
        mx, den, acc = _online(score, av, mx, den, acc)
    for s in range(0, W, 2):
        t0 = tl.load(I + (h * M + qb) * W + s)
        t1 = tl.load(I + (h * M + qb) * W + s + 1, s + 1 < W, 0)
        r = tl.where(lane < 64, t0 * 64 + lane, t1 * 64 + lane - 64)
        good = (lane < 64) | (s + 1 < W)
        k = tl.load(K + (h * N + r[None, :]) * 128 + d[:, None], good[None, :], 0)
        v = tl.load(V + (h * N + r[:, None]) * 128 + d[None, :], good[:, None], 0)
        score = tl.dot(q, k) * (0.08838834764831845 * 1.4426950408889634)
        score = tl.where(good[None, :], score, -float('inf'))
        mx, den, acc = _online(score, v, mx, den, acc)
    tl.store(O + (h * N + rows[:, None]) * 128 + d[None, :], (acc / den[:, None]).to(tl.bfloat16))

def attention(q, k, v, ids, route, ak, av, lm, bm=64):
    h, n, d = q.shape
    m = ids.shape[1]
    w = ids.shape[2]
    out = torch.empty_like(q)
    _spark[m * (64 // bm), h](q, k, v, ids, route, ak, av, lm, out, n, m, w, bm)
    import torch_npu
    y = torch_npu.npu_fusion_attention(q[:, m * 64:][None], k[None], v[None], h, input_layout='BNSD', scale=128 ** (-0.5), keep_prob=1.0, sparse_mode=0)[0]
    out[:, m * 64:].copy_(y[0])
    return out
attention_gather_baseline = attention

def attention(*args, **kwargs):
    import os
    if _settings.get('ATTN', 'cann') == 'gather':
        return attention_gather_baseline(*args, **kwargs)
    if _settings.get('ATTN', 'cann') == 'cann_joint':
        from .attention_cann_joint import attention as impl
    elif _settings.get('ATTN') == 'cann_fused':
        from .attention_fused_reweight import attention as impl
    elif _settings.get('ATTN', 'cann') == 'cann':
        from .attention_cann import attention as impl
    else:
        from .attention_blockptr import attention as impl
    return impl(*args, **kwargs)
