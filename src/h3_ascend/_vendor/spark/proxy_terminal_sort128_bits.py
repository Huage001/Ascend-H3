
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl
from .proxy_fused import _unit_row

@triton.jit
def _proxy_split(center, unit, distance, weight, left_cap, right_cap, LANDMARKS: tl.constexpr, ITERATIONS: tl.constexpr):
    """One capacity-weighted binary proxy split; returns direction and child weights.

    ``center``/``unit`` are [K, D] FP32, ``distance`` is [K, K] cosine
    distance, ``weight`` is the [K] int32 active landmark weight at this node.
    The sorted-prefix capacity rule of ``_cosine_proxy_node`` is evaluated
    through a rank comparison, which is the same total order ``(delta, id)``.
    """
    landmark = tl.arange(0, LANDMARKS)
    first_slot = landmark[:, None]
    second_slot = landmark[None, :]
    pair_valid = (first_slot < second_slot) & (weight[:, None] > 0) & (weight[None, :] > 0)
    masked = tl.where(pair_valid, distance, -float('inf'))
    maximum_distance = tl.max(tl.max(masked, axis=1), axis=0)
    pair_index = first_slot * LANDMARKS + second_slot
    candidate = tl.where(masked == maximum_distance, pair_index, LANDMARKS * LANDMARKS)
    farthest = tl.min(tl.min(candidate, axis=1), axis=0)
    first = farthest // LANDMARKS
    second = farthest % LANDMARKS
    active_count = tl.sum((weight > 0).to(tl.int32), axis=0)
    only = tl.argmax((weight > 0).to(tl.int32), axis=0, tie_break_left=True)
    first = tl.where(active_count < 2, only, first)
    second = tl.where(active_count < 2, only, second)
    left_center = tl.sum(tl.where(landmark[:, None] == first, center, 0.0), axis=0)
    right_center = tl.sum(tl.where(landmark[:, None] == second, center, 0.0), axis=0)
    left_weight = tl.zeros((LANDMARKS,), tl.int32)
    right_weight = weight
    for _ in tl.static_range(ITERATIONS):
        direction = _unit_row(right_center) - _unit_row(left_center)
        delta = tl.sum(unit * direction[None, :], axis=1)
        padded = tl.cat(delta, tl.full((32,), float('inf'), tl.float32), can_reorder=True)
        padded = tl.cat(padded, tl.full((64,), float('inf'), tl.float32), can_reorder=True)
        ordered = tl.sort(padded, descending=False)
        cut = tl.sum(tl.where(tl.arange(0, 128) == 15, ordered, 0.0), 0)
        below = delta < cut
        equal = delta == cut
        need = 16 - tl.sum(below.to(tl.int32), 0)
        allbits = tl.sum(tl.where(equal, tl.full((32,), 1, tl.uint32) << landmark.to(tl.uint32), 0).to(tl.uint32), 0)
        prefix = allbits & tl.full((32,), 4294967295, tl.uint32) >> (31 - landmark).to(tl.uint32)
        prefix = prefix - (prefix >> 1 & 1431655765)
        prefix = (prefix & 858993459) + (prefix >> 2 & 858993459)
        prefix = prefix + (prefix >> 4) & 252645135
        prefix = prefix * 16843009 >> 24
        selected = below | equal & (prefix <= need)
        left_weight = tl.where(selected, 4, 0)
        right_weight = 4 - left_weight
        left_center = tl.sum(center * left_weight[:, None].to(tl.float32), axis=0) / left_cap
        right_center = tl.sum(center * right_weight[:, None].to(tl.float32), axis=0) / right_cap
    direction = _unit_row(right_center) - _unit_row(left_center)
    return (direction, left_weight, right_weight)

@triton.jit
def _kernel(C, D, O, B: tl.constexpr):
    p = tl.program_id(0)
    i = tl.arange(0, 32)
    d = tl.arange(0, 128)
    c = tl.load(C + (p * 32 + i[:, None]) * 128 + d[None, :]).to(tl.float32)
    u = c / tl.maximum(tl.sqrt(tl.sum(c * c, 1)), 1e-12)[:, None]
    dist = tl.load(D + (p * 32 + i[:, None]) * 32 + i[None, :])
    w = tl.full((32,), 4, tl.int32)
    direction, lw, rw = _proxy_split(c, u, dist, w, 64, 64, 32, 2)
    tl.store(O + p * 128 + d, direction)

def directions(c, weights, caps, bp=2):
    assert caps == (64, 64)
    u = c.float()
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dist = (1 - torch.bmm(u, u.transpose(1, 2))).contiguous()
    out = torch.empty((c.shape[0], 1, 128), device=c.device, dtype=torch.float32)
    _kernel[c.shape[0],](c, dist, out, c.shape[0])
    return out
