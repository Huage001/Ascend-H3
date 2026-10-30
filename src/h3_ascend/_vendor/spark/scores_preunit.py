
"""Normalize every immutable tree feature once, retaining the exact FP16 operand."""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl

@triton.jit
def _pack(X, Y, N: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, 128)
    x = tl.load(X + r[:, None] * 128 + d[None, :], r[:, None] < N, 0).to(tl.float32)
    norm = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), 1e-12)
    tl.store(Y + r[:, None] * 128 + d[None, :], (x / norm[:, None]).to(tl.float16), r[:, None] < N)

@triton.jit
def _scores(X, I, D, O, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr, COL: tl.constexpr):
    b = tl.program_id(1)
    r = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, 128)
    c = tl.arange(0, 16)
    ix = tl.load(I + b * N + r, r < N, 0)
    x = tl.load(X + ix[:, None] * 128 + d[None, :], r[:, None] < N, 0)
    dirs = tl.load(D + (b * S + c[:, None]) * 128 + d[None, :], c[:, None] < S, 0).to(tl.float16)
    v = tl.dot(x, dirs.T)
    if COL:
        tl.store(O + (b * S + c[None, :]) * N + r[:, None], v, (r[:, None] < N) & (c[None, :] < S))
    else:
        tl.store(O + (b * N + r[:, None]) * S + c[None, :], v, (r[:, None] < N) & (c[None, :] < S))

def pack(x):
    y = torch.empty_like(x, dtype=torch.float16)
    _pack[triton.cdiv(x.shape[0], 128),](x, y, x.shape[0], 128)
    return y

def scores(x, ids, dirs):
    import os
    col = _settings.get('SCORE_LAYOUT') == 'column'
    b, n = ids.shape
    s = dirs.shape[1]
    out = torch.empty((b, s, n) if col else (b, n, s), device=x.device, dtype=torch.float32)
    tile = 128 if n <= 128 else int(_settings.get('PREUNIT_BM', '128'))
    assert tile in (128, 256, 512)
    _scores[triton.cdiv(n, tile), b](x, ids, dirs, out, n, s, tile, col)
    return out.transpose(1, 2) if col else out

@triton.jit
def _vector(X, I, D, O, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    b = tl.program_id(1)
    r = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, 128)
    ix = tl.load(I + b * N + r, r < N, 0)
    x = tl.load(X + ix[:, None] * 128 + d[None, :], r[:, None] < N, 0).to(tl.float32)
    for j in tl.static_range(S):
        direction = tl.load(D + (b * S + j) * 128 + d).to(tl.float16).to(tl.float32)
        val = tl.sum(x * direction[None, :], 1)
        tl.store(O + (b * S + j) * N + r, val, r < N)
_old_scores = scores

def scores(x, ids, dirs):
    import os
    if _settings.get('SCORE_VECTOR_SMALL') != '1' or dirs.shape[1] > 3:
        return _old_scores(x, ids, dirs)
    b, n = ids.shape
    s = dirs.shape[1]
    out = torch.empty((b, s, n), device=x.device, dtype=torch.float32)
    _vector[triton.cdiv(n, 128), b](x, ids, dirs, out, n, s, 128)
    return out.transpose(1, 2)

@triton.jit
def _guard_terminal(X, I, D, O, STAT):
    b = tl.program_id(0)
    r = tl.arange(0, 128)
    d = tl.arange(0, 128)
    ix = tl.load(I + b * 128 + r)
    xh = tl.load(X + ix[:, None] * 128 + d[None, :])
    dh = tl.load(D + b * 128 + d).to(tl.float16)
    prod = xh.to(tl.float32) * dh[None, :].to(tl.float32)
    val = tl.sum(prod, 1)
    ordered = tl.sort(val, descending=False)
    lo = tl.sum(tl.where(r == 63, ordered, 0.0), 0)
    hi = tl.sum(tl.where(r == 64, ordered, 0.0), 0)
    bound = 3e-05 * tl.max(tl.sum(tl.abs(prod), 1), 0) + 1e-07
    fallback = hi - lo <= 2 * bound
    tl.store(O + b * 128 + r, val)
    tl.store(STAT + b, fallback.to(tl.int32))

@triton.jit
def _fallback_terminal(X, I, D, O, STAT):
    b = tl.program_id(0)
    fallback = tl.load(STAT + b)
    if fallback:
        r = tl.arange(0, 128)
        d = tl.arange(0, 128)
        col = tl.arange(0, 16)
        ix = tl.load(I + b * 128 + r)
        x = tl.load(X + ix[:, None] * 128 + d[None, :])
        dirs = tl.load(D + (b + col[:, None]) * 128 + d[None, :], col[:, None] == 0, 0).to(tl.float16)
        mat = tl.dot(x, dirs.T)
        val = tl.sum(tl.where(col[None, :] == 0, mat, 0.0), 1)
        tl.store(O + b * 128 + r, val)

@triton.jit
def _gate_terminal(D, O, STAT):
    b = tl.program_id(0)
    r = tl.arange(0, 128)
    val = tl.load(O + b * 128 + r)
    ordered = tl.sort(val, descending=False)
    lo = tl.sum(tl.where(r == 63, ordered, 0.0), 0)
    hi = tl.sum(tl.where(r == 64, ordered, 0.0), 0)
    d = tl.load(D + b * 128 + r).to(tl.float16).to(tl.float32)
    bound = 3e-05 * 1.002 * tl.sqrt(tl.sum(d * d, 0)) + 1e-07
    tl.store(STAT + b, (hi - lo <= 2 * bound).to(tl.int32))
_previous_scores = scores
LAST_STAT = None

def scores(x, ids, dirs):
    import os
    global LAST_STAT
    if _settings.get('SCORE_VECTOR_SMALL') != 'guard' or dirs.shape[1] != 1 or ids.shape[1] != 128:
        return _previous_scores(x, ids, dirs)
    b = ids.shape[0]
    out = torch.empty((b, 1, 128), device=x.device, dtype=torch.float32)
    stat = torch.empty((b,), device=x.device, dtype=torch.int32)
    _vector[1, b](x, ids, dirs, out, 128, 1, 128)
    _gate_terminal[b,](dirs, out, stat)
    _fallback_terminal[b,](x, ids, dirs, out, stat)
    LAST_STAT = stat
    return out.transpose(1, 2)

@triton.jit
def _pack_norm(X, Y, V, N: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, 128)
    x = tl.load(X + r[:, None] * 128 + d[None, :], r[:, None] < N, 0).to(tl.float32)
    sq = tl.sum(x * x, 1)
    tl.store(V + r, sq, r < N)
    norm = tl.maximum(tl.sqrt(sq), 1e-12)
    tl.store(Y + r[:, None] * 128 + d[None, :], (x / norm[:, None]).to(tl.float16), r[:, None] < N)

def pack_norm(x):
    y = torch.empty_like(x, dtype=torch.float16)
    v = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    _pack_norm[triton.cdiv(x.shape[0], 128),](x, y, v, x.shape[0], 128)
    return (y, v)

@triton.jit
def _scores_parent_pair(X, I, D, O, N: tl.constexpr, S: tl.constexpr, P: tl.constexpr, COL: tl.constexpr):
    group = tl.program_id(1) * 2
    tile = tl.program_id(0) * 128
    r = tl.arange(0, 256)
    d = tl.arange(0, 128)
    c = tl.arange(0, 16)
    parent = group + r // 128
    pos = tile + r % 128
    ix = tl.load(I + parent * N + pos, (parent < P) & (pos < N), 0)
    x = tl.load(X + ix[:, None] * 128 + d[None, :], ((parent < P) & (pos < N))[:, None], 0)
    dp = group + c // S
    dc = c % S
    direction = tl.load(D + (dp[:, None] * S + dc[:, None]) * 128 + d[None, :], ((c < 2 * S) & (dp < P))[:, None], 0).to(tl.float16)
    val = tl.dot(x, direction.T)
    for j in tl.static_range(S):
        chosen = tl.gather(val, (r // 128 * S + j)[:, None], axis=1).reshape(256)
        valid = (parent < P) & (pos < N)
        if COL:
            tl.store(O + (parent * S + j) * N + pos, chosen, valid)
        else:
            tl.store(O + (parent * N + pos) * S + j, chosen, valid)
_unpaired_scores = scores

def scores(x, ids, dirs):
    import os
    b, n = ids.shape
    s = dirs.shape[1]
    pair = _settings.get('PREUNIT_PAIR')
    if pair not in ('small', 'terminal') or s > 3 or (pair == 'terminal' and n != 128):
        return _unpaired_scores(x, ids, dirs)
    col = _settings.get('SCORE_LAYOUT') == 'column'
    out = torch.empty((b, s, n) if col else (b, n, s), device=x.device, dtype=torch.float32)
    _scores_parent_pair[triton.cdiv(n, 128), triton.cdiv(b, 2)](x, ids, dirs, out, n, s, b, col)
    return out.transpose(1, 2) if col else out
_indirect_scores = scores

def scores(x, ids, dirs):
    import os
    mode = _settings.get('PREUNIT_DENSE')
    if mode == 'row':
        return _importlib.import_module('h3_ascend._vendor.spark.scores_dense').row(x, ids, dirs)
    if mode == 'pair' and ids.shape[1] == 128:
        return _importlib.import_module('h3_ascend._vendor.spark.scores_dense').pair(x, ids, dirs)
    use = mode == 'all' or (mode == 'nonterminal' and ids.shape[1] > 128) or (mode == 'small' and dirs.shape[1] <= 3) or (mode == 'terminal' and ids.shape[1] == 128)
    if use:
        return _importlib.import_module('h3_ascend._vendor.spark.scores_dense').run(x, ids, dirs)
    return _indirect_scores(x, ids, dirs)
