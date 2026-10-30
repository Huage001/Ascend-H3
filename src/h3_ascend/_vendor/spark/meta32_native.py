
"""Exact int32 tree metadata; no changes to feature or score arithmetic."""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import os
import triton.language as tl
from .common import ROOT
_loaded = False

def load():
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_meta32/build/libspark_meta32.so'))
        _loaded = True

@triton.jit
def _gather(S, P, O, N: tl.constexpr, SB: tl.constexpr, SN: tl.constexpr, PB: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(1)
    i = tl.program_id(0) * B + tl.arange(0, B)
    pos = tl.load(P + row * PB + i, i < N, 0)
    v = tl.load(S + row * SB + pos * SN, i < N, 0)
    tl.store(O + row * N + i, v, i < N)

def gather(scores, pos):
    if _settings.get('META32_GATHER') == 'torch':
        return scores.gather(1, pos.long())
    b, n = pos.shape
    out = torch.empty((b, n), device=scores.device, dtype=scores.dtype)
    _gather[triton.cdiv(n, 512), b](scores, pos, out, n, scores.stride(0), scores.stride(1), pos.stride(0), 512)
    return out

def partition(ids, pos, scores, take, debug=False):
    if _settings.get('META_FAST') == '1':
        return _importlib.import_module('h3_ascend._vendor.spark.partition_fast_native').partition(ids, pos, scores, take, debug)
    load()
    scores = scores.contiguous()
    cuts = scores.sort(-1, descending=False).values[:, take - 1].contiguous()
    oi, op, status = torch.ops.spark_meta32.partition(ids, pos, scores, cuts, take)
    if debug:
        good = torch.tensor([take, ids.shape[1] - take, 0, 0, 0, 0, 0, 0], device=ids.device, dtype=torch.int32)
        assert bool((status == good).all()), status.cpu()
    return (oi, op, status) if debug else (oi, op)

def root_partition(norm, active):
    b, n = norm.shape
    rem = n - active
    np = (n + 63) // 64 * 64
    score = torch.nn.functional.pad(-norm, (0, np - n), value=float('inf'))
    ids = torch.arange(np, device=norm.device, dtype=torch.int32)[None].expand(b, -1)
    order, _ = partition(ids, ids, score, rem)
    return (order[:, rem:rem + active], order[:, :rem])

def terminal(scores, ids, heads, starts, out):
    load()
    return torch.ops.spark_meta32.partition_write(scores.contiguous(), ids, heads.contiguous(), starts.contiguous(), out)
