
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl

@triton.jit
def _chol(A, L):
    b = tl.program_id(0)
    i = tl.arange(0, 128)
    jj = tl.arange(0, 16)
    for c in range(128):
        col = tl.load(A + b * 16384 + c * 128 + i)
        diag = tl.sum(tl.where(i == c, col, 0.0), 0)
        col = col / tl.sqrt(diag)
        tl.store(L + b * 16384 + c * 128 + i, col, i >= c)
        for start in range(c // 16, 8):
            j = start * 16 + jj
            right = tl.gather(col, j, 0)
            ptr = A + b * 16384 + j[:, None] * 128 + i[None, :]
            a = tl.load(ptr)
            a = tl.where((i[None, :] > c) & (j[:, None] > c), a - col[None, :] * right[:, None], a)
            tl.store(ptr, a)
        tl.debug_barrier()

def cholesky(a):
    work = a.transpose(1, 2).contiguous()
    out = torch.zeros_like(a)
    _chol[a.shape[0],](work, out, enable_auto_bind_sub_block=False)
    return out.transpose(1, 2)
