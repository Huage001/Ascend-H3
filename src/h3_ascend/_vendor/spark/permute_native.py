
import importlib as _importlib
from . import settings as _settings
import torch
from .common import ROOT
_loaded = False

def kv(k, v, ids, g, pad=64):
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_permute/build/libspark_permute.so'))
        _loaded = True
    return torch.ops.spark_permute.permute(k, v, ids.contiguous(), g, pad)
