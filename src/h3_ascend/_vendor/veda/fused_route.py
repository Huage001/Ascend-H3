
"""Fuse mask preparation/assembly; retain Torch's official FP32 topk."""
import importlib as _importlib
import sys
from pathlib import Path
import torch
import triton
import triton.language as tl
from h3_ascend._vendor.miowtion.veda import mask

@triton.jit
def _prepare(S, VALID, O, N: tl.constexpr, V: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    h = row // V
    r = row % V
    c = tl.arange(0, B)
    valid = tl.load(VALID + c, c < V, 0)
    value = tl.load(S + (h * N + r) * N + c, c < V, 0)
    value = tl.where(valid, value, float('-inf'))
    value = tl.where(c == r, float('inf'), value)
    tl.store(O + row * V + c, value, c < V)

@triton.jit
def _assemble(IDX, VAL, EXTRA, VALID, O, N: tl.constexpr, V: tl.constexpr, K: tl.constexpr, LO: tl.constexpr, B: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    h = row // N
    r = row % N
    c = tl.arange(0, B)
    valid = tl.load(VALID + c, c < N, 0)
    initial = valid & ((r >= V) | (c >= V))
    tl.store(O + row * N + c, initial, c < N)
    tl.debug_barrier()
    if r < V:
        j = tl.arange(0, BK)
        idx = tl.load(IDX + (h * V + r) * K + j, j < K, 0)
        vals = tl.load(VAL + (h * V + r) * K + j, j < K, float('-inf'))
        extra = tl.load(EXTRA + r)
        keep = (j < LO + extra.to(tl.int32)) & (vals > float('-inf'))
        tl.store(O + row * N + idx, keep, j < K)

class Route:

    def __init__(self, layout, blocks):
        assert len(blocks) == 1 and blocks[0].start == 0 and (blocks[0].stop == layout.n_video_tiles)
        block = blocks[0]
        self.layout = layout
        self.lo, self.hi, frac = mask.split_budget(block.budget.per_row(block.real_tokens, layout.n_video_tiles), layout.n_video_tiles)
        self.extra = mask.bresenham_extra(layout.n_video_tiles, frac, layout.valid_count.device)

    def __call__(self, scores):
        layout = self.layout
        n = layout.n_tiles
        v = layout.n_video_tiles
        h = scores.shape[0]
        assert scores.is_contiguous() and scores.dtype == torch.float32 and (scores.shape == (h, n, n))
        prepared = torch.empty((h, v, v), device=scores.device, dtype=scores.dtype)
        _prepare[h * v,](scores, layout.kv_ok, prepared, n, v, triton.next_power_of_2(v))
        vals, indices = torch.topk(prepared, self.hi, dim=-1, sorted=True)
        out = torch.empty((h, n, n), device=scores.device, dtype=torch.bool)
        _assemble[h * n,](indices, vals, self.extra, layout.kv_ok, out, n, v, self.hi, self.lo, triton.next_power_of_2(n), triton.next_power_of_2(self.hi))
        return out
