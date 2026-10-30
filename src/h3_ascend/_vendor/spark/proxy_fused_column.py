# Derived from Spark-H3 2eb2dc1; upstream license retained in official snapshot.
import importlib as _importlib
from . import settings as _settings
import torch
import triton
import triton.language as tl

@triton.jit
def _unit_row(vector):
    """FP32 unit vector with the proxy kernel's clamp."""
    return vector / tl.maximum(tl.sqrt(tl.sum(vector * vector, 0)), 1e-12)

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
        before = (delta[:, None] < delta[None, :]) | (delta[:, None] == delta[None, :]) & (first_slot < second_slot)
        prefix_before = tl.sum(tl.where(before, weight[:, None], 0), axis=0)
        take = tl.minimum(tl.maximum(left_cap - prefix_before, 0), weight)
        left_weight = take
        right_weight = weight - take
        left_center = tl.sum(center * left_weight[:, None].to(tl.float32), axis=0) / left_cap
        right_center = tl.sum(center * right_weight[:, None].to(tl.float32), axis=0) / right_cap
    direction = _unit_row(right_center) - _unit_row(left_center)
    return (direction, left_weight, right_weight)

@triton.jit
def _fused_proxy_directions_kernel(centers, root_weight, pair_dist, LC, RC, directions, LANDMARKS: tl.constexpr, INTERNAL: tl.constexpr, SLOTS: tl.constexpr):
    parent = tl.program_id(0)
    landmark = tl.arange(0, LANDMARKS)
    dims = tl.arange(0, 128)
    center = tl.load(centers + (parent * LANDMARKS + landmark[:, None]) * 128 + dims[None, :]).to(tl.float32)
    unit = center / tl.maximum(tl.sqrt(tl.sum(center * center, axis=1)), 1e-12)[:, None]
    distance = tl.load(pair_dist + (parent * LANDMARKS + landmark[:, None]) * LANDMARKS + landmark[None, :])
    slot_rows = tl.arange(0, SLOTS)
    root = tl.load(root_weight + landmark).to(tl.int32)
    weight_nodes = tl.where((slot_rows == 0)[:, None], root[None, :], 0)
    for node in range(INTERNAL):
        direction, left_weight, right_weight = _proxy_split(center, unit, distance, tl.sum(tl.where((slot_rows == node)[:, None], weight_nodes, 0), axis=0), tl.load(LC + node), tl.load(RC + node), LANDMARKS, 2)
        weight_nodes = tl.where((slot_rows == 2 * node + 1)[:, None], left_weight[None, :], weight_nodes)
        weight_nodes = tl.where((slot_rows == 2 * node + 2)[:, None], right_weight[None, :], weight_nodes)
        tl.store(directions + (parent * INTERNAL + node) * 128 + dims, direction)

def topology(caps):
    ranges = [(0, len(caps), 0)]
    out = []
    slot = 0
    while ranges:
        following = []
        for start, end, node in ranges:
            mid = (start + end) // 2
            out.append((node, sum(caps[start:mid]), sum(caps[mid:end])))
            if mid - start > 1:
                following.append((start, mid, 2 * node + 1))
            if end - mid > 1:
                following.append((mid, end, 2 * node + 2))
        ranges = following
    return tuple(out)
_CAP_CACHE = {}

def directions(centers, weights, caps):
    if _settings.get('PROXY_DIST') == 'raw_bf16':
        dist = _importlib.import_module('h3_ascend._vendor.spark.proxy_dist').raw(centers)
    else:
        u = centers.float()
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        dist = (1 - torch.bmm(u, u.transpose(1, 2))).contiguous()
    top = topology(caps)
    out = torch.empty((centers.shape[0], len(top), 128), device=centers.device, dtype=torch.float32)
    key = (tuple(caps), str(centers.device))
    if key not in _CAP_CACHE:
        _CAP_CACHE[key] = [torch.tensor([t[i] for t in top], device=centers.device, dtype=torch.int32) for i in (1, 2)]
    lc, rc = _CAP_CACHE[key]
    _fused_proxy_directions_kernel[centers.shape[0],](centers, weights[0].contiguous().int(), dist, lc, rc, out, LANDMARKS=32, INTERNAL=len(top), SLOTS=triton.next_power_of_2(2 * len(top) + 1))
    return out
