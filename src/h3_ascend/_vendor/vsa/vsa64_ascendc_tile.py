
"""TND gather plus native joint pooling without unused FP32 correction Q."""
import importlib as _importlib
import torch
import triton
import triton.language as tl

@triton.jit
def _gather(Q, K, V, G, S, QT, KT, VT, JOINT, QS: tl.constexpr, QH: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    component = tl.program_id(2)
    r = tl.arange(0, 64)
    dim = tl.arange(0, D)
    size = tl.load(S + tile)
    src = tl.load(G + tile * 64 + r)
    if component == 0:
        val = tl.load(Q + src[:, None] * QS + head * QH + dim[None, :], r[:, None] < size, 0)
    elif component == 1:
        val = tl.load(K + src[:, None] * KS + head * KH + dim[None, :], r[:, None] < size, 0)
    else:
        val = tl.load(V + src[:, None] * VS + head * VH + dim[None, :], r[:, None] < size, 0)
    val = tl.where(r[:, None] < size, val, 0)
    tpos = (tile * 64 + r[:, None]) * H * D + head * D + dim[None, :]
    jpos = (tile * 64 + r[:, None]) * 3 * H * D + component * H * D + head * D + dim[None, :]
    if component == 0:
        tl.store(QT + tpos, val)
    elif component == 1:
        tl.store(KT + tpos, val)
    else:
        tl.store(VT + tpos, val)
    tl.store(JOINT + jpos, val)

def tile_pool(q, k, v, gather, sizes):
    n, h, d = (sizes.numel(), q.shape[1], q.shape[2])
    t = [torch.empty((n * 64, h, d), dtype=q.dtype, device=q.device) for _ in range(3)]
    joint = torch.empty((n, 64, 3, h, d), dtype=q.dtype, device=q.device)
    _gather[n, h, 3](q, k, v, gather, sizes, *t, joint, *q.stride()[:2], *k.stride()[:2], *v.stride()[:2], n, h, d)
    pool = (joint.sum(1, dtype=torch.float32) / sizes[:, None, None, None]).permute(1, 2, 0, 3)
    return (*t, *pool.unbind(0))
