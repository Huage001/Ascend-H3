
import importlib as _importlib
from . import settings as _settings
import torch
import os
from .common import ROOT
_loaded = set()
_masks = {}

def directions(centers, weights, caps, bp=2):
    assert len(caps) == 4 and centers.shape[1:] == (32, 128) and (sum(caps) % 32 == 0)
    ops = _settings.get('PROXY_FOUR_OPS', 'ascendc_proxy_four')
    namespace = 'spark_' + ops.removeprefix('ascendc_')
    lib = str(ROOT / 'ops' / ops / _settings.get('PROXY_FOUR_BUILD', 'build') / ('lib' + namespace + '.so'))
    if lib not in _loaded:
        torch.ops.load_library(lib)
        _loaded.add(lib)
    key = str(centers.device)
    if key not in _masks:
        bits = [4294967295 << i + 1 & 4294967295 if i < 31 else 0 for i in range(32)]
        _masks[key] = torch.tensor([v if v < 2 ** 31 else v - 2 ** 32 for v in bits], device=centers.device, dtype=torch.int32)
    u = centers.float()
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dist = (1 - torch.bmm(u, u.transpose(1, 2))).contiguous()
    return getattr(torch.ops, namespace).directions(centers, dist, _masks[key], *caps)
