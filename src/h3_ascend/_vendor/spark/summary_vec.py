
"""Vector-core weighted summary preserving BF16 probability before PV reduction."""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl

@triton.jit
def _summary(A, K, V, AK, AV, LM, N: tl.constexpr, M: tl.constexpr):
    kb = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 128)
    kp = tl.make_block_ptr(K + h * N * 128, (N, 128), (128, 1), (kb * 64, 0), (64, 128), (1, 0))
    vp = tl.make_block_ptr(V + h * N * 128, (N, 128), (128, 1), (kb * 64, 0), (64, 128), (1, 0))
    k = tl.load(kp).to(tl.float32)
    v = tl.load(vp).to(tl.float32)
    a = tl.load(A + h * 128 + d).to(tl.float32)
    s = tl.sum(k * a[None, :], 1) * (0.08838834764831845 * 1.4426950408889634)
    mx = tl.max(s, 0)
    p = tl.exp2(s - mx)
    den = tl.sum(p, 0)
    pb = p.to(tl.bfloat16).to(tl.float32)
    ak = tl.sum(k * pb[:, None], 0) / den
    av = tl.sum(v * pb[:, None], 0) / den
    stored = ak.to(tl.bfloat16)
    lm = (mx + tl.log2(den)) * 0.6931471805599453 - tl.sum(a * stored.to(tl.float32), 0) * 0.08838834764831845
    tl.store(AK + (h * M + kb) * 128 + d, stored)
    tl.store(AV + (h * M + kb) * 128 + d, av.to(tl.bfloat16))
    tl.store(LM + h * M + kb, lm)

def summaries(a, k, v, m):
    h, n, d = k.shape
    ak = torch.empty((h, m, 128), device=k.device, dtype=k.dtype)
    av = torch.empty_like(ak)
    lm = torch.empty((h, m), device=k.device, dtype=torch.float32)
    _summary[m, h](a, k, v, ak, av, lm, n, m)
    return (ak, av, lm)
