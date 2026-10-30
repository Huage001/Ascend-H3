
"""Exact stable smallest16 membership for finite FP32 rows; padded Int32 IDs."""
import importlib as _importlib
from . import settings as _settings
import torch
import os
from .common import ROOT
_loaded = False

def run(scores, debug=False):
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_select32' / _settings.get('SELECT32_BUILD', 'build') / 'libspark_select32.so'))
        _loaded = True
    order, status = torch.ops.spark_select32.select(scores.contiguous())
    if debug:
        good = torch.tensor([16, 0, 0, 0, 0, 0, 0, 0], device=scores.device, dtype=torch.int32)
        assert bool((status == good).all()), status.cpu()
    return order
