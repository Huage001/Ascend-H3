
"""One exact byte-selection pass replacing where + full FP32 padding copy."""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import os
import triton.language as tl

@triton.jit
def _prepare(R, L, O, H: tl.constexpr, M: tl.constexpr, MP: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, NEG: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.arange(0, BK)
    head = rows // M
    valid = (rows < H * M)[:, None] & (c < M)[None, :]
    route = tl.load(R + rows[:, None] * M + c[None, :], valid, 1)
    lm = tl.load(L + head[:, None] * M + c[None, :], valid, 0)
    value = tl.where((route == 0) & valid, lm, tl.full((BM, BK), NEG, tl.float32))
    tl.store(O + rows[:, None] * MP + c[None, :], value, (rows < H * M)[:, None] & (c < MP)[None, :])

def prepare(route, lm, mp, neg, bm=None):
    h, m, _ = route.shape
    bm = bm or int(_settings.get('BIAS_ROWS', '2'))
    out = torch.empty((h, m, mp), device=route.device, dtype=torch.float32)
    _prepare[triton.cdiv(h * m, bm),](route, lm, out, h, m, mp, bm, triton.next_power_of_2(mp), neg)
    return out
