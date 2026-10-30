
"""Terminal equal-weight proxy held entirely in CANN on-chip buffers."""
import importlib as _importlib
from . import settings as _settings
import torch
import os
from .common import ROOT
_loaded = set()
_masks = {}

def run(centers, debug=False):
    global _loaded
    ops = _settings.get('PROXY_ONCHIP_OPS', 'ascendc_proxy_onchip')
    build = _settings.get('PROXY_ONCHIP_BUILD', 'build')
    lib = str(ROOT / 'ops' / ops / build / 'libspark_proxy_onchip.so')
    if lib not in _loaded:
        torch.ops.load_library(lib)
        _loaded.add(lib)
    native = getattr(torch.ops, 'spark_' + ops.removeprefix('ascendc_'))
    key = str(centers.device)
    if key not in _masks:
        bits = [4294967295 << i + 1 & 4294967295 if i < 31 else 0 for i in range(32)]
        bits = [v if v < 2 ** 31 else v - 2 ** 32 for v in bits]
        _masks[key] = torch.tensor(bits, device=centers.device, dtype=torch.int32)
    u = centers.float()
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dist = (1 - torch.bmm(u, u.transpose(1, 2))).contiguous()
    return native.directions(centers, dist, _masks[key], debug, int(_settings.get('PROXY_NORM_LR', '0'))) if _settings.get('PROXY_ONCHIP_BUILD', 'build') != 'build' else native.directions(centers, dist, _masks[key], debug)

def directions(centers, weights, caps, bp=2):
    assert caps == (64, 64) and centers.shape[1:] == (32, 128)
    out, unit = run(centers, False)
    return out
