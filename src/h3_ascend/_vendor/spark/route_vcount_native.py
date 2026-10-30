
import importlib as _importlib
from . import settings as _settings
import torch
import os
from .common import ROOT
_loaded = False

def route(scores, w):
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_route_vcount' / _settings.get('ROUTE_VCOUNT_BUILD', 'build') / 'libspark_route_vcount.so'))
        _loaded = True
    h, m, _ = scores.shape
    r = torch.ops.spark_route_vcount.route(scores.contiguous(), w)[..., :m].contiguous()
    ids = torch.empty(1, device=scores.device, dtype=torch.int32).expand(h, m, w)
    return (ids, r)
