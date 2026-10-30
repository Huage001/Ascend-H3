
"""One CANN exact call including dense tail queries; summary merge in-place."""
import importlib as _importlib
from . import settings as _settings
import os
import time
import torch
import triton
from . import attention_cann as base
_CACHE = {}

def attention(q, k, v, ids, route, ak, av, lm, bm=64, real_tokens=None):
    global LAST_PHASES
    prof = _settings.get('PROFILE') == '1'
    phase = {}
    if prof:
        torch.npu.synchronize()
        started = time.perf_counter()
    if not base._LOADED:
        __import__('h3_ascend.native',fromlist=['load']).load('vsa_paired64')
        base._LOADED = True
    h, np, d = q.shape
    m = route.shape[1]
    a = m * 64
    nt = np // 64
    n = real_tokens or np
    bk = int(_settings.get('MERGE_BK', '128'))
    mp = triton.cdiv(m, bk) * bk
    key = (np, n, str(q.device))
    if key not in _CACHE:
        sizes = torch.full((nt,), 64, device=q.device, dtype=torch.int32)
        sizes[-1] = n - (nt - 1) * 64
        _CACHE[key] = sizes
    sizes = _CACHE[key]
    mask = torch.ones((1, h, nt, nt), device=q.device, dtype=torch.int8)
    mask[0, :, :m, :m].copy_(route)
    if prof:
        torch.npu.synchronize()
        now = time.perf_counter()
        phase['exact_prepare'] = (now - started) * 1000
        started = now
    eo, el, _ = base.exact_op()(q[None], k[None], v[None], mask, sizes, [64, 64], None, None, 'BNSD', h, 128 ** (-0.5), 0, True, 128, int(_settings.get('STACK', '1')))
    if prof:
        torch.npu.synchronize()
        now = time.perf_counter()
        phase['exact_including_dense_context'] = (now - started) * 1000
        started = now
    akt = torch.nn.functional.pad(ak.transpose(1, 2), (0, mp - m))
    avp = torch.nn.functional.pad(av, (0, 0, 0, mp - m))
    if prof:
        torch.npu.synchronize()
        now = time.perf_counter()
        phase['reweight_layout'] = (now - started) * 1000
        started = now
    base._merge[m * (64 // bm), h](q, akt, avp, lm, route, eo, el, eo, a, np, m, mp, bm, np, bk)
    if prof:
        torch.npu.synchronize()
        phase['reweight_merge'] = (time.perf_counter() - started) * 1000
        LAST_PHASES = phase
    return eo[0]
