
import importlib as _importlib
from . import settings as _settings
import torch
from .common import ROOT
_loaded = False

def partition(pos, score, take):
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(ROOT / 'ops/ascendc_partition_pos/build/libspark_partition_pos.so'))
        _loaded = True
    score = score.contiguous()
    cut = score.sort(dim=-1).values[:, take - 1].contiguous()
    return torch.ops.spark_partition_pos.partition(pos, score, cut, take)

def split(ids, scores, caps):
    n = ids.shape[1]
    pos = torch.arange(n, device=ids.device)[None].expand(ids.shape[0], -1)
    front = [(pos, 0, 0, len(caps))]
    out = []
    while front:
        p, node, s, e = front.pop(0)
        mid = (s + e) // 2
        take = sum(caps[s:mid])
        v = scores[:, :, node].gather(1, p)
        op = partition(p, v, take)
        for c, a, b, ch in [(op[:, :take], s, mid, node * 2 + 1), (op[:, take:], mid, e, node * 2 + 2)]:
            if b - a == 1:
                out.append((a, c))
            else:
                front.append((c, ch, a, b))
    positions = torch.cat([v for _, v in sorted(out)], 1)
    return ids.gather(1, positions)
