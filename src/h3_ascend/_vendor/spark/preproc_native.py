
import importlib as _importlib
from . import settings as _settings
import torch
import torch_npu
import os
from .common import ROOT
_loaded = False
_kc_loaded = False

def load():
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_preproc' / _settings.get('PREPROC_BUILD', 'build-v11') / 'libspark_preproc.so'))
        _loaded = True

def centroids(k, m):
    global _kc_loaded
    if not _kc_loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_preproc/build/libspark_preproc.so'))
        _kc_loaded = True
    return torch.ops.spark_preproc_v4.centroids(k, m)

def partition(cur, pos, score, take, debug=False):
    load()
    score = score.contiguous()
    cut = score.sort(dim=-1, descending=False).values[:, take - 1].contiguous()
    oi, op, status = torch.ops.spark_preproc_v4.partition(cur, pos, score, cut, take)
    if debug:
        expected = torch.zeros(status.shape[1], device=cur.device, dtype=torch.int32)
        expected[0] = take
        expected[1] = cur.shape[1] - take
        assert bool((status == expected).all()), status.cpu()
    return (oi, op, status) if debug else (oi, op)

def root_partition(norm, active):
    b, n = norm.shape
    rem = n - active
    np = (n + 63) // 64 * 64
    score = torch.nn.functional.pad(-norm, (0, np - n), value=float('inf'))
    ids = torch.arange(np, device=norm.device)[None].expand(b, -1)
    order, _ = partition(ids, ids, score, rem)
    return (order[:, rem:rem + active], order[:, :rem])

def route(scores, w):
    if _settings.get('ROUTE_HYBRID') == '1':
        return _importlib.import_module('h3_ascend._vendor.spark.route_hybrid_native').route(scores, w)
    if _settings.get('ROUTE_RADIX') == '1':
        return _importlib.import_module('h3_ascend._vendor.spark.route_radix_native').route(scores, w)
    if _settings.get('ROUTE_VCOUNT') == '1':
        return _importlib.import_module('h3_ascend._vendor.spark.route_vcount_native').route(scores, w)
    if _settings.get('ROUTE_CACHE') == '1':
        return _importlib.import_module('h3_ascend._vendor.spark.route_cache_native').route(scores, w)
    if _settings.get('ROUTE_FAST') == '1':
        return _importlib.import_module('h3_ascend._vendor.spark.route_fast_native').route(scores, w)
    load()
    h, m, _ = scores.shape
    r = torch.ops.spark_preproc_v4.route(scores.contiguous(), w)[..., :m].contiguous()
    ids = torch.empty(1, device=scores.device, dtype=torch.int32).expand(h, m, w)
    return (ids, r)
