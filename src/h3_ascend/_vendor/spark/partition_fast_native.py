
"""Exact stable partition with no-boundary-tie fast path and alias-aware payload."""
import importlib as _importlib
from . import settings as _settings
import torch
import os
from .common import ROOT
_loaded = False

def partition(ids, pos, scores, take, debug=False):
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_partition_fast' / _settings.get('PARTITION_FAST_BUILD', 'build') / 'libspark_partition_fast.so'))
        _loaded = True
    scores = scores.contiguous()
    ordered = scores.sort(dim=-1, descending=False).values
    cuts = ordered[:, take - 1:take + 1].contiguous()
    a, b, status = torch.ops.spark_partition_fast.partition(ids, pos, scores, cuts, take)
    if debug:
        expected = torch.tensor([take, ids.shape[1] - take, 0, 0, 0, 0, 0, 0], device=ids.device, dtype=torch.int32)
        assert bool((status == expected).all()), status.cpu()
    return (a, b, status) if debug else (a, b)
