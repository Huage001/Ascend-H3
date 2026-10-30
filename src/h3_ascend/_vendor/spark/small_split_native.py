
import importlib as _importlib
from . import settings as _settings
import torch
from .common import ROOT
_loaded = False

def split(ids, scores, caps):
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_small_split/build/libspark_small_split.so'))
        _loaded = True
    ix = torch.ops.spark_small_split.split(scores.transpose(1, 2).contiguous(), sum(caps[:2]), caps[0], caps[2])
    return ids.gather(1, ix.long())
