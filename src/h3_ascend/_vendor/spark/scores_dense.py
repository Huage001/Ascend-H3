
"""Materialize exact normalized rows through native IndexSelect, then dense QK."""
import importlib as _importlib
from . import settings as _settings
import os
import torch
import triton
import triton.language as tl

@triton.jit
def _dense(X, D, O, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr, COL: tl.constexpr):
    p = tl.program_id(1)
    r = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, 128)
    c = tl.arange(0, 16)
    x = tl.load(X + (p * N + r[:, None]) * 128 + d[None, :], r[:, None] < N, 0)
    direction = tl.load(D + (p * S + c[:, None]) * 128 + d[None, :], c[:, None] < S, 0).to(tl.float16)
    y = tl.dot(x, direction.T)
    if COL:
        tl.store(O + (p * S + c[None, :]) * N + r[:, None], y, (r[:, None] < N) & (c[None, :] < S))
    else:
        tl.store(O + (p * N + r[:, None]) * S + c[None, :], y, (r[:, None] < N) & (c[None, :] < S))

def run(x, ids, dirs):
    b, n = ids.shape
    s = dirs.shape[1]
    col = _settings.get('SCORE_LAYOUT') == 'column'
    dense = x.index_select(0, ids.reshape(-1))
    out = torch.empty((b, s, n) if col else (b, n, s), device=x.device, dtype=torch.float32)
    tile = 128 if n <= 128 else int(_settings.get('DENSE_BM', '256'))
    _dense[triton.cdiv(n, tile), b](dense, dirs, out, n, s, tile, col)
    return out.transpose(1, 2) if col else out

@triton.jit
def _pair_dense(X, D, O):
    group = tl.program_id(0)
    r = tl.arange(0, 256)
    d = tl.arange(0, 128)
    c = tl.arange(0, 16)
    x = tl.load(X + (group * 256 + r[:, None]) * 128 + d[None, :])
    direction = tl.load(D + (group * 2 + c[:, None]) * 128 + d[None, :], c[:, None] < 2, 0).to(tl.float16)
    v = tl.dot(x, direction.T)
    chosen = tl.gather(v, (r // 128)[:, None], axis=1).reshape(256)
    tl.store(O + group * 256 + r, chosen)

def pair(x, ids, dirs):
    b, n = ids.shape
    assert n == 128 and dirs.shape[1] == 1 and (b % 2 == 0)
    dense = x.index_select(0, ids.reshape(-1))
    out = torch.empty((b, n, 1), device=x.device, dtype=torch.float32)
    _pair_dense[b // 2,](dense, dirs, out)
    return out

@triton.jit
def _dense_row(X, D, O, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    p = tl.program_id(1)
    r = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, 128)
    c = tl.arange(0, 16)
    x = tl.load(X + (p * N + r[:, None]) * 128 + d[None, :], r[:, None] < N, 0)
    direction = tl.load(D + (p * S + c[:, None]) * 128 + d[None, :], c[:, None] < S, 0).to(tl.float16)
    val = tl.dot(x, direction.T)
    tl.store(O + (p * N + r[:, None]) * 16 + c[None, :], val, r[:, None] < N)

def row(x, ids, dirs):
    b, n = ids.shape
    s = dirs.shape[1]
    dense = x.index_select(0, ids.reshape(-1))
    out = torch.empty((b, n, 16), device=x.device, dtype=torch.float32)
    tile = 128 if n <= 128 else int(_settings.get('DENSE_BM', '512'))
    _dense_row[triton.cdiv(n, tile), b](dense, dirs, out, n, s, tile)
    return out[:, :, :s]

def root(x, b, dirs):
    """Project original contiguous root rows; split keeps original row coordinates."""
    assert x.shape[0] % b == 0 and dirs.shape[0] == b
    n = x.shape[0] // b
    s = dirs.shape[1]
    col = _settings.get('SCORE_LAYOUT') == 'column'
    out = torch.empty((b, s, n) if col else (b, n, s), device=x.device, dtype=torch.float32)
    _dense[triton.cdiv(n, 512), b](x, dirs, out, n, s, 512, col)
    return out.transpose(1, 2) if col else out
