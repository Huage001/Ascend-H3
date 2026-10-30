
"""One CANN online-softmax stream over exact KV and weighted summaries."""
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import os
from .common import ROOT
_ops = {}

def op():
    engine = _settings.get('FUSED_ENGINE', 'v3')
    build = _settings.get('PIPE_BUILD', 'build') if engine == 'pipe' else _settings.get('V4_BUILD', 'build') if engine == 'v4' else _settings.get('FUSED_BUILD', 'build-v3')
    key = (engine, build)
    if key not in _ops:
        if engine == 'pipe':
            torch.ops.load_library(str(ROOT / 'ops/ascendc_spark_joint_pipe' / build / 'libspark_joint_pipe.so'))
            _ops[key] = torch.ops.spark_joint_pipe_ext.forward
        elif engine == 'v4':
            torch.ops.load_library(str(ROOT / 'ops/ascendc_spark_joint_v4' / build / 'libspark_joint_v4.so'))
            _ops[key] = torch.ops.spark_joint_v4_ext.forward
        else:
            torch.ops.load_library(str(ROOT / 'ops/ascendc_spark_joint' / build / 'libspark_joint.so'))
            _ops[key] = torch.ops.spark_joint_ext.forward
    return _ops[key]

def attention(q, k, v, ids, route, ak, av, lm, bm=64, real_tokens=None, prepacked=None):
    h, n, d = q.shape
    m = route.shape[1]
    nt = n // 64
    mp = triton.cdiv(m, 64) * 64
    real = real_tokens or n
    neg = float(_settings.get('FUSED_BIAS_NEG', '-3.402823466e38'))
    if _settings.get('BIAS_PREP') == 'triton':
        bias = _importlib.import_module('h3_ascend._vendor.spark.bias_prepare').prepare(route, lm, mp, neg)
    else:
        bias = torch.nn.functional.pad(torch.where(route != 0, torch.full((), neg, device=q.device), lm[:, None, :]), (0, mp - m), value=neg).contiguous()
    if _settings.get('FUSED_TEST_BIAS') == 'zero':
        bias = torch.zeros_like(bias)
    if _settings.get('KV_TAIL_INPLACE') == '1':
        assert k.shape[1] == n + mp and v.shape[1] == n + mp
        kp = k
        vp = v
        kp[:, n:n + m].copy_(ak)
        vp[:, n:n + m].copy_(av)
    elif prepacked is not None:
        kp, vp = _importlib.import_module('h3_ascend._vendor.spark.attention_overlap').finish_pack(prepacked, ak, av, n, m)
    elif _settings.get('CACHE_KV') == '1':
        kp, vp = _importlib.import_module('h3_ascend._vendor.spark.attention_workspace').packed_kv(q, k, v, ak, av, m, mp, real)
    else:
        kp = torch.cat((k, torch.nn.functional.pad(ak, (0, 0, 0, mp - m))), 1).contiguous()
        vp = torch.cat((v, torch.nn.functional.pad(av, (0, 0, 0, mp - m))), 1).contiguous()
    nkt = kp.shape[1] // 64
    if _settings.get('CACHE_CONSTANTS') == '1':
        mask, sizes = _importlib.import_module('h3_ascend._vendor.spark.attention_workspace').constants(q, m, mp, real)
        mask[0, :, :m, :m].copy_(route)
    else:
        mask = torch.ones((1, h, nt, nkt), device=q.device, dtype=torch.int8)
        mask[0, :, :m, :m].copy_(route)
        mask[:, :, :, nt:].zero_()
        mask[:, :, :m, nt:].fill_(1)
        sizes = torch.full((nkt,), 64, device=q.device, dtype=torch.int32)
        sizes[nt - 1] = real - (nt - 1) * 64
        sizes[-1] = m - (mp // 64 - 1) * 64
    y, lse, _ = op()(q[None], kp[None], vp[None], mask, sizes, [64, 64], None, None, 'BNSD', h, 128 ** (-0.5), 0, _settings.get('FUSED_RETURN_LSE', '1') != '0', int(_settings.get('FUSED_COMPUTE_TILE', '128')), int(_settings.get('STACK', '4')), None if _settings.get('FUSED_TEST_BIAS') == 'disabled' else bias, n, m, mp)
    return y[0]
