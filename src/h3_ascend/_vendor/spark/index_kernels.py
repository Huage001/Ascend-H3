
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl

@triton.jit
def _write(C, H, S, O, N: tl.constexpr, CS: tl.constexpr):
    p = tl.program_id(0)
    r = tl.arange(0, 64)
    h = tl.load(H + p)
    s = tl.load(S + p)
    value = tl.load(C + p * CS + r)
    tl.store(O + h * N + s * 64 + r, value)

def write(child, heads, starts, out):
    _write[child.shape[0],](child, heads, starts, out, out.shape[1], child.stride(0))
