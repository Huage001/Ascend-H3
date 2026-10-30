
"""CANN paired KV64 exact attention + one-loop query-wise summary/LSE merge.
Default Spark selected sets, weighted summaries, and exact context are retained.
Intermediate exact output is BF16, an extra rounding versus the CUDA fused path.
"""
import importlib as _importlib
from . import settings as _settings
import os
import torch
import torch_npu
import triton
import triton.language as tl
_LOADED = False
_PAIR_LOADED = False
_COMPACT_LOADED = False

def exact_op():
    global _LOADED, _PAIR_LOADED, _COMPACT_LOADED
    if _settings.get('SPARK_COMPACT') == '1':
        if not _COMPACT_LOADED:
            from .common import ROOT
            torch.ops.load_library(str(ROOT / 'ops/ascendc_spark_compact/build/libspark_compact.so'))
            _COMPACT_LOADED = True
        return torch.ops.spark_compact_ext.forward
    if _settings.get('SPARK_PAIR') == '1':
        if not _PAIR_LOADED:
            from .common import ROOT
            torch.ops.load_library(str(ROOT / 'ops/ascendc_spark_pair/build/libspark_pair.so'))
            _PAIR_LOADED = True
        return torch.ops.spark_pair_ext.forward
    if not _LOADED:
        __import__('h3_ascend.native',fromlist=['load']).load('vsa_paired64')
        _LOADED = True
    return torch.ops.sol_h3_bsa_ext.forward

@triton.jit
def _merge(Q, AK, AV, LM, R, EO, EL, O, A: tl.constexpr, N: tl.constexpr, M: tl.constexpr, MP: tl.constexpr, BM: tl.constexpr, QS: tl.constexpr, BK: tl.constexpr=128):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    start = pid * BM
    qb = start // 64
    rows = start + tl.arange(0, BM)
    lane = tl.arange(0, 128)
    qp = tl.make_block_ptr(Q + h * QS * 128, (QS, 128), (128, 1), (start, 0), (BM, 128), (1, 0))
    q = tl.load(qp)
    mx = tl.full((BM,), -float('inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 128), tl.float32)
    for s in range(0, M, BK):
        kp = tl.make_block_ptr(AK + h * MP * 128, (128, MP), (MP, 1), (0, s), (128, BK), (1, 0))
        vp = tl.make_block_ptr(AV + h * MP * 128, (MP, 128), (128, 1), (s, 0), (BK, 128), (1, 0))
        ak = tl.load(kp)
        av = tl.load(vp)
        r = s + tl.arange(0, BK)
        lm = tl.load(LM + h * M + r, r < M, 0)
        sel = tl.load(R + (h * M + qb) * M + r, r < M, 1)
        score = tl.dot(q, ak) * (0.08838834764831845 * 1.4426950408889634) + lm[None, :] * 1.4426950408889634
        if BM == 128:
            sel2 = tl.load(R + (h * M + qb + 1) * M + r, (r < M) & (qb + 1 < M), 1)
            selected = tl.where((tl.arange(0, BM) < 64)[:, None], sel[None, :], sel2[None, :])
            score = tl.where((r < M)[None, :] & (selected == 0), score, -float('inf'))
        else:
            score = tl.where(((r < M) & (sel == 0))[None, :], score, -float('inf'))
        nm = tl.maximum(mx, tl.max(score, 1))
        safe = tl.where(nm == -float('inf'), 0.0, nm)
        alpha = tl.exp2(mx - safe)
        pr = tl.exp2(score - safe[:, None])
        acc = acc * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), av)
        den = den * alpha + tl.sum(pr, 1)
        mx = nm
    coarse = acc / den[:, None]
    lc = (mx + tl.log2(den)) * 0.6931471805599453
    le = tl.load(EL + h * QS + rows)
    maxi = tl.maximum(lc, le)
    wc = tl.exp(lc - maxi)
    we = tl.exp(le - maxi)
    ep = tl.make_block_ptr(EO + h * QS * 128, (QS, 128), (128, 1), (start, 0), (BM, 128), (1, 0))
    exact = tl.load(ep).to(tl.float32)
    out = (exact * we[:, None] + coarse * wc[:, None]) / (we + wc)[:, None]
    op = tl.make_block_ptr(O + h * N * 128, (A, 128), (128, 1), (start, 0), (BM, 128), (1, 0))
    tl.store(op, out.to(tl.bfloat16), boundary_check=(0,))

def attention(q, k, v, ids, route, ak, av, lm, bm=64):
    global _LOADED
    if not _LOADED:
        __import__('h3_ascend.native',fromlist=['load']).load('vsa_paired64')
        _LOADED = True
    h, n, d = q.shape
    m = ids.shape[1]
    a = m * 64
    np = triton.cdiv(n, 64) * 64
    nt = np // 64
    mp = triton.cdiv(m, 128) * 128
    query = q[:, :a].contiguous()[None]
    kp = torch.nn.functional.pad(k, (0, 0, 0, np - n))[None]
    vp = torch.nn.functional.pad(v, (0, 0, 0, np - n))[None]
    mask = torch.ones((1, h, m, nt), device=q.device, dtype=torch.int8)
    mask[0, :, :, :m].copy_(route)
    sizes = torch.full((nt,), 64, device=q.device, dtype=torch.int32)
    sizes[-1] = n - (nt - 1) * 64
    eo, el, _ = exact_op()(query, kp, vp, mask, sizes, [64, 64], None, None, 'BNSD', h, 128 ** (-0.5), 0, True, 128, int(_settings.get('STACK', '1')))
    if _settings.get('DIAG') == '1':
        global LAST_EXACT
        LAST_EXACT = (eo, el)
    if _settings.get('DEBUG_SYNC') == '1':
        torch.npu.synchronize()
        print('CANN_EXACT_OK', flush=True)
    akt = torch.nn.functional.pad(ak.transpose(1, 2), (0, mp - m))
    avp = torch.nn.functional.pad(av, (0, 0, 0, mp - m))
    out = torch.empty_like(q)
    _merge[m * (64 // bm), h](query, akt, avp, lm, route, eo, el, out, a, n, m, mp, bm, a)
    if _settings.get('DEBUG_SYNC') == '1':
        torch.npu.synchronize()
        print('COARSE_MERGE_OK', flush=True)
    tail = q[:, a:].contiguous()[None]
    dense = torch_npu.npu_fused_infer_attention_score(tail, k[None], v[None], num_heads=h, num_key_value_heads=h, input_layout='BNSD', scale=128 ** (-0.5), actual_seq_lengths=[n - a], actual_seq_lengths_kv=[n])[0]
    out[:, a:].copy_(dense[0])
    return out
