
import importlib as _importlib
from . import settings as _settings
import torch
from .common import ROOT
_loaded = False

def kv(k, v, ids, g, tail):
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_permute_tail/build/libspark_permute_tail.so'))
        _loaded = True
    return torch.ops.spark_permute_tail.permute(k, v, ids.contiguous(), g, tail)
