
"""Explicit producer events for independent copies and vector preparation."""
import importlib as _importlib
from . import settings as _settings
import torch
_streams = {}

def stream():
    key = str(torch.npu.current_device())
    if key not in _streams:
        _streams[key] = torch.npu.Stream()
    return _streams[key]

def raw_pack(q, k, v, m, real):
    from . import attention_workspace as ws
    mp = (m + 63) // 64 * 64
    key = ws.key(q, m, real)
    h, n, d = q.shape
    if key not in ws._kv:
        ws._kv[key] = (torch.zeros((h, n + mp, d), device=q.device, dtype=q.dtype), torch.zeros((h, n + mp, d), device=q.device, dtype=q.dtype))
    kp, vp = ws._kv[key]
    s = stream()
    s.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(s):
        kp[:, :n].copy_(k)
        vp[:, :n].copy_(v)
        event = torch.npu.Event()
        event.record(s)
    return (kp, vp, event)

def finish_pack(packed, ak, av, n, m):
    kp, vp, event = packed
    torch.npu.current_stream().wait_event(event)
    kp[:, n:n + m].copy_(ak)
    vp[:, n:n + m].copy_(av)
    return (kp, vp)

def kv_permute(k, v, ki, g, pad):
    import os
    from . import kernels
    from . import permute_fused
    s = stream()
    s.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(s):
        if _settings.get('PERMUTE_FUSED') in ('kv', 'qkv'):
            kp, vp = permute_fused.kv(k, v, ki, g, pad)
        else:
            kp = kernels.permute(k, ki, g, pad=pad)
            vp = kernels.permute(v, ki, g, pad=pad)
        event = torch.npu.Event()
        event.record(s)
    return (kp, vp, event)

def summary(anchor, k, v, m):
    from . import kernels
    from . import summary_vec
    import os
    s = stream()
    s.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(s):
        out = summary_vec.summaries(anchor, k, v, m) if _settings.get('SUMMARY') == 'vec' else kernels.summaries(anchor, k, v, m)
        event = torch.npu.Event()
        event.record(s)
    return (out, event)
