
"""Bit-copy-only joint permutations: KV shares the token index load."""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl

@triton.jit
def _kv(K, V, I, OK, OV, N: tl.constexpr, G: tl.constexpr, H: tl.constexpr, T: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    h = tl.program_id(1)
    r = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, 128)
    idx = tl.load(I + h * N + tl.minimum(r, N - 1), r < N, 0)
    src = tl.where(r < N, G + idx, r - N)
    a = tl.load(K + (src[:, None] * H + h) * 128 + d[None, :], r[:, None] < T, 0)
    b = tl.load(V + (src[:, None] * H + h) * 128 + d[None, :], r[:, None] < T, 0)
    tl.store(OK + (h * P + r[:, None]) * 128 + d[None, :], a, r[:, None] < P)
    tl.store(OV + (h * P + r[:, None]) * 128 + d[None, :], b, r[:, None] < P)

def kv(k, v, ids, g, pad=64):
    h, n = ids.shape
    t = n + g
    p = triton.cdiv(t, pad) * pad if pad else t
    b = int(_settings.get('PERMUTE_B', '32'))
    ko = torch.empty((h, p, 128), device=k.device, dtype=k.dtype)
    vo = torch.empty_like(ko)
    _kv[triton.cdiv(p, b), h](k, v, ids, ko, vo, n, g, h, t, p, b)
    return (ko, vo)

@triton.jit
def _qkv(Q, K, V, QI, KI, OQ, OK, OV, N: tl.constexpr, G: tl.constexpr, H: tl.constexpr, T: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    h = tl.program_id(1)
    r = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, 128)
    qi = tl.load(QI + h * N + tl.minimum(r, N - 1), r < N, 0)
    ki = tl.load(KI + h * N + tl.minimum(r, N - 1), r < N, 0)
    qs = tl.where(r < N, G + qi, r - N)
    ks = tl.where(r < N, G + ki, r - N)
    q = tl.load(Q + (qs[:, None] * H + h) * 128 + d[None, :], r[:, None] < T, 0)
    k = tl.load(K + (ks[:, None] * H + h) * 128 + d[None, :], r[:, None] < T, 0)
    v = tl.load(V + (ks[:, None] * H + h) * 128 + d[None, :], r[:, None] < T, 0)
    dest = (h * P + r[:, None]) * 128 + d[None, :]
    mask = r[:, None] < P
    tl.store(OQ + dest, q, mask)
    tl.store(OK + dest, k, mask)
    tl.store(OV + dest, v, mask)

def qkv(q, k, v, qi, ki, g, pad=64):
    h, n = qi.shape
    t = n + g
    p = triton.cdiv(t, pad) * pad if pad else t
    b = int(_settings.get('PERMUTE_B', '32'))
    qo = torch.empty((h, p, 128), device=q.device, dtype=q.dtype)
    ko = torch.empty_like(qo)
    vo = torch.empty_like(qo)
    _qkv[triton.cdiv(p, b), h](q, k, v, qi, ki, qo, ko, vo, n, g, h, t, p, b)
    return (qo, ko, vo)
