
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl

@triton.jit
def _transform(Q, K, F, O, H: tl.constexpr, N: tl.constexpr, G: tl.constexpr):
    b = tl.program_id(1)
    h = b % H
    r = tl.program_id(0) * 64
    if b < H:
        xp = tl.make_block_ptr(K + (G * H + h) * 128, (N, 128), (H * 128, 1), (r, 0), (64, 128), (1, 0))
        x = tl.load(xp, boundary_check=(0,), padding_option='zero')
    else:
        xp = tl.make_block_ptr(Q + (G * H + h) * 128, (N, 128), (H * 128, 1), (r, 0), (64, 128), (1, 0))
        x = tl.load(xp, boundary_check=(0,), padding_option='zero')
    fp = tl.make_block_ptr(F + b * 16384, (128, 128), (128, 1), (0, 0), (128, 128), (1, 0))
    f = tl.load(fp)
    y = tl.dot(x, f).to(tl.bfloat16)
    op = tl.make_block_ptr(O + b * N * 128, (N, 128), (128, 1), (r, 0), (64, 128), (1, 0))
    tl.store(op, y, boundary_check=(0,))

def transform(q, k, factors, n, g):
    h = q.shape[2]
    out = torch.empty((2 * h, n, 128), device=q.device, dtype=torch.bfloat16)
    f = factors.to(torch.bfloat16).contiguous()
    _transform[triton.cdiv(n, 64), 2 * h](q, k, f, out, h, n, g)
    return out

@triton.jit
def _norm(X, O, N: tl.constexpr):
    h = tl.program_id(1)
    r = tl.program_id(0) * 32 + tl.arange(0, 32)
    d = tl.arange(0, 128)
    p = tl.make_block_ptr(X + h * N * 128, (N, 128), (128, 1), (tl.program_id(0) * 32, 0), (32, 128), (1, 0))
    x = tl.load(p, boundary_check=(0,), padding_option='zero').to(tl.float32)
    y = tl.sum(x * x, 1)
    tl.store(O + h * N + r, y, r < N)

def norm(x):
    h, n, d = x.shape
    out = torch.empty((h, n), device=x.device, dtype=torch.float32)
    _norm[triton.cdiv(n, 32), h](x, out, n)
    return out

@triton.jit
def _qpool(Q, QM, PART, N: tl.constexpr, M: tl.constexpr):
    kb = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 128)
    ptr = tl.make_block_ptr(Q + h * N * 128, (N, 128), (128, 1), (kb * 64, 0), (64, 128), (1, 0))
    q = tl.load(ptr).to(tl.float32)
    sums = tl.sum(q, 0)
    tl.store(QM + (h * M + kb) * 128 + d, (sums / 64.0).to(tl.bfloat16))
    tl.store(PART + (h * M + kb) * 128 + d, sums)

def query_pool(q, m):
    h, n, d = q.shape
    qm = torch.empty((h, m, 128), device=q.device, dtype=torch.bfloat16)
    part = torch.empty((h, m, 128), device=q.device, dtype=torch.float32)
    _qpool[m, h](q, qm, part, n, m)
    anchor = (part.sum(1) / (m * 64)).to(torch.bfloat16)
    return (qm, anchor)
