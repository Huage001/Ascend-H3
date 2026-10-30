
import importlib as _importlib
from . import settings as _settings
import torch
import os
from .common import ROOT
_loaded = False

def inverse(x, ids, g):
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_inverse/build/libspark_inverse.so'))
        _loaded = True
    return torch.ops.spark_inverse.inverse(x, ids.contiguous(), g, int(_settings.get('INVERSE_B', '128')))
