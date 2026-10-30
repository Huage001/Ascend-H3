
"""Per-stream inference workspaces; dynamic routing is always copied afresh."""
import importlib as _importlib
from . import settings as _settings
import torch
_constants = {}
_kv = {}

def key(q, m, real):
    stream = torch.npu.current_stream()
    handle = getattr(stream, 'npu_stream', None)
    if handle is None:
        handle = str(stream)
    return (str(q.device), str(q.dtype), handle, q.shape[0], q.shape[1], q.shape[2], m, real)

def constants(q, m, mp, real):
    h, n, d = q.shape
    nt = n // 64
    nkt = nt + mp // 64
    k = key(q, m, real)
    if k not in _constants:
        mask = torch.ones((1, h, nt, nkt), device=q.device, dtype=torch.int8)
        mask[:, :, :, nt:].zero_()
        mask[:, :, :m, nt:].fill_(1)
        sizes = torch.full((nkt,), 64, device=q.device, dtype=torch.int32)
        sizes[nt - 1] = real - (nt - 1) * 64
        sizes[-1] = m - (mp // 64 - 1) * 64
        _constants[k] = (mask, sizes)
    return _constants[k]

def packed_kv(q, k, v, ak, av, m, mp, real):
    h, n, d = q.shape
    cachekey = key(q, m, real)
    if cachekey not in _kv:
        kp = torch.zeros((h, n + mp, d), device=q.device, dtype=q.dtype)
        vp = torch.zeros_like(kp)
        _kv[cachekey] = (kp, vp)
    kp, vp = _kv[cachekey]
    kp[:, :n].copy_(k)
    vp[:, :n].copy_(v)
    kp[:, n:n + m].copy_(ak)
    vp[:, n:n + m].copy_(av)
    return (kp, vp)
