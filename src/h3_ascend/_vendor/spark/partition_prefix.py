
"""Exact-capacity stable binary partition: inverse rank + prefix + scatter.
All original token IDs are preserved; this only replaces the index sorting work.
"""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl

@triton.jit
def _flags(SORT, MASK, N: tl.constexpr, TAKE: tl.constexpr, B: tl.constexpr):
    h = tl.program_id(1)
    r = tl.program_id(0) * B + tl.arange(0, B)
    pos = tl.load(SORT + h * N + r, r < N, 0).to(tl.int32)
    tl.store(MASK + h * N + pos, (r < TAKE).to(tl.int32), r < N)

@triton.jit
def _scatter(IDS, POS, MASK, PREFIX, OI, OP, N: tl.constexpr, TAKE: tl.constexpr, B: tl.constexpr, IS0: tl.constexpr, IS1: tl.constexpr, PS0: tl.constexpr, PS1: tl.constexpr):
    h = tl.program_id(1)
    r = tl.program_id(0) * B + tl.arange(0, B)
    mask = tl.load(MASK + h * N + r, r < N, 0)
    rank = tl.load(PREFIX + h * N + r, r < N, 0)
    dest = tl.where(mask != 0, rank - 1, TAKE + r - rank)
    ids = tl.load(IDS + h * IS0 + r * IS1, r < N, 0)
    pos = tl.load(POS + h * PS0 + r * PS1, r < N, 0)
    tl.store(OI + h * N + dest, ids, r < N)
    tl.store(OP + h * N + dest, pos, r < N)

def partition(cur, pos, score, take):
    order = score.argsort(dim=1, stable=True)
    b, n = cur.shape
    flag = torch.empty((b, n), device=cur.device, dtype=torch.int32)
    _flags[triton.cdiv(n, 512), b](order, flag, n, take, 512)
    prefix = flag.cumsum(1, dtype=torch.int32)
    oi = torch.empty_like(cur)
    op = torch.empty_like(pos)
    _scatter[triton.cdiv(n, 512), b](cur, pos, flag, prefix, oi, op, n, take, 512, *cur.stride(), *pos.stride())
    return (oi, op)
