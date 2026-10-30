
import importlib as _importlib
from . import settings as _settings
import torch
from .common import ROOT
_loaded = False

def run(scores, ids, heads, starts, out):
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_terminal/build/libspark_terminal.so'))
        _loaded = True
    return torch.ops.spark_terminal.partition_write(scores.contiguous(), ids, heads.contiguous(), starts.contiguous(), out)
