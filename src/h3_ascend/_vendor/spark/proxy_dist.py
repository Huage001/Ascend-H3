
"""Algebraic BF16 Gram candidate; distinct FP32 rounding, validate routes explicitly."""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl

@triton.jit
def _gram(C, O):
    p = tl.program_id(0)
    i = tl.arange(0, 32)
    d = tl.arange(0, 128)
    x = tl.load(C + (p * 32 + i[:, None]) * 128 + d[None, :])
    raw = tl.dot(x, tl.trans(x))
    tl.store(O + (p * 32 + i[:, None]) * 32 + i[None, :], raw)

@triton.jit
def _normalize(C, R, O):
    p = tl.program_id(0)
    i = tl.arange(0, 32)
    d = tl.arange(0, 128)
    x = tl.load(C + (p * 32 + i[:, None]) * 128 + d[None, :]).to(tl.float32)
    norm = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), 1e-12)
    raw = tl.load(R + (p * 32 + i[:, None]) * 32 + i[None, :])
    dist = 1.0 - raw / (norm[:, None] * norm[None, :])
    tl.store(O + (p * 32 + i[:, None]) * 32 + i[None, :], dist)

def raw(centers):
    gram = torch.empty((centers.shape[0], 32, 32), device=centers.device, dtype=torch.float32)
    out = torch.empty_like(gram)
    _gram[centers.shape[0],](centers, gram)
    _normalize[centers.shape[0],](centers, gram, out)
    return out
