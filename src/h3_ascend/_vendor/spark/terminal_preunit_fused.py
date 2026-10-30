
"""Cached exact FP16 operands permit QK/threshold/stable partition/write fusion."""
import importlib as _importlib
from . import settings as _settings
import triton
import triton.language as tl

@triton.jit
def _kernel(X, Z, D, H, S, O, N: tl.constexpr):
    p = tl.program_id(0)
    r = tl.arange(0, 128)
    d = tl.arange(0, 128)
    c = tl.arange(0, 16)
    h = tl.load(H + p)
    start = tl.load(S + p)
    ids = tl.load(Z + p * 128 + r)
    x = tl.load(X + (h * N + ids[:, None]) * 128 + d[None, :])
    direction = tl.load(D + (p + c[:, None]) * 128 + d[None, :], c[:, None] == 0, 0).to(tl.float16)
    mat = tl.dot(x, direction.T)
    v = tl.sum(tl.where(c[None, :] == 0, mat, 0.0), 1)
    ordered = tl.sort(v, descending=False)
    cut = tl.sum(tl.where(r == 63, ordered, 0.0), 0)
    below = v < cut
    eq = v == cut
    need = 64 - tl.sum(below.to(tl.int32), 0)
    left = below | eq & (tl.cumsum(eq.to(tl.int32), 0) <= need)
    prefix = tl.cumsum(left.to(tl.int32), 0)
    dest = tl.where(left, prefix - 1, 64 + r - prefix)
    tl.store(O + h * N + start * 64 + dest, ids)

def run(source, ids, dirs, heads, starts, out):
    _kernel[ids.shape[0],](source, ids, dirs, heads, starts, out, out.shape[1])
