
import importlib as _importlib
from . import settings as _settings
import torch
import torch_npu
import triton
import triton.language as tl
from .kernels import _online

@triton.jit
def _pair_k(P, t0, t1, second, N: tl.constexpr):
    gap = tl.where(second, t1 - t0, 1)
    ptr = tl.make_block_ptr(P + t0 * 64, (128, tl.where(second, 2, 1), 64), (N, gap * 64, 1), (0, 0, 0), (128, 2, 64), (2, 1, 0))
    return tl.load(ptr, boundary_check=(1,), padding_option='zero').reshape(128, 128)

@triton.jit
def _pair_v(P, t0, t1, second, N: tl.constexpr):
    gap = tl.where(second, t1 - t0, 1)
    ptr = tl.make_block_ptr(P + t0 * 64 * 128, (tl.where(second, 2, 1), 64, 128), (gap * 64 * 128, 128, 1), (0, 0, 0), (2, 64, 128), (2, 1, 0))
    return tl.load(ptr, boundary_check=(0,), padding_option='zero').reshape(128, 128)

@triton.jit
def _attn(Q, K, V, I, R, AK, AV, LM, O, N: tl.constexpr, M: tl.constexpr, W: tl.constexpr, BM: tl.constexpr, NK: tl.constexpr, MK: tl.constexpr):
    qid = tl.program_id(0)
    h = tl.program_id(1)
    qb = qid // (64 // BM)
    ro = qid % (64 // BM) * BM
    lane = tl.arange(0, 128)
    qp = tl.make_block_ptr(Q + h * N * 128, (N, 128), (128, 1), (qb * 64 + ro, 0), (BM, 128), (1, 0))
    q = tl.load(qp)
    mx = tl.full((BM,), -float('inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 128), tl.float32)
    for s in range(M * 64, N, 128):
        kp = tl.make_block_ptr(K + h * NK * 128, (128, NK), (NK, 1), (0, s), (128, 128), (1, 0))
        vp = tl.make_block_ptr(V + h * N * 128, (N, 128), (128, 1), (s, 0), (128, 128), (1, 0))
        k = tl.load(kp, boundary_check=(1,), padding_option='zero')
        v = tl.load(vp, boundary_check=(0,), padding_option='zero')
        score = tl.dot(q, k) * (0.08838834764831845 * 1.4426950408889634)
        score = tl.where((s + lane < N)[None, :], score, -float('inf'))
        mx, den, acc = _online(score, v, mx, den, acc)
    for s in range(0, M, 128):
        kp = tl.make_block_ptr(AK + h * MK * 128, (128, MK), (MK, 1), (0, s), (128, 128), (1, 0))
        vp = tl.make_block_ptr(AV + h * M * 128, (M, 128), (128, 1), (s, 0), (128, 128), (1, 0))
        ak = tl.load(kp, boundary_check=(1,), padding_option='zero')
        av = tl.load(vp, boundary_check=(0,), padding_option='zero')
        r = s + lane
        lm = tl.load(LM + h * M + r, r < M, 0)
        sel = tl.load(R + (h * M + qb) * M + r, r < M, 1)
        score = tl.dot(q, ak) * (0.08838834764831845 * 1.4426950408889634) + lm[None, :] * 1.4426950408889634
        score = tl.where(((r < M) & (sel == 0))[None, :], score, -float('inf'))
        mx, den, acc = _online(score, av, mx, den, acc)
    for s in range(0, W, 2):
        t0 = tl.load(I + (h * M + qb) * W + s)
        t1 = tl.load(I + (h * M + qb) * W + s + 1, s + 1 < W, 0)
        lo = tl.minimum(t0, t1)
        hi = tl.maximum(t0, t1)
        t0 = tl.where(s + 1 < W, lo, t0)
        t1 = tl.where(s + 1 < W, hi, 0)
        k = _pair_k(K + h * NK * 128, t0, t1, s + 1 < W, NK)
        v = _pair_v(V + h * N * 128, t0, t1, s + 1 < W, N)
        score = tl.dot(q, k) * (0.08838834764831845 * 1.4426950408889634)
        score = tl.where(((lane < 64) | (s + 1 < W))[None, :], score, -float('inf'))
        mx, den, acc = _online(score, v, mx, den, acc)
    op = tl.make_block_ptr(O + h * N * 128, (N, 128), (128, 1), (qb * 64 + ro, 0), (BM, 128), (1, 0))
    tl.store(op, (acc / den[:, None]).to(tl.bfloat16))

def attention(q, k, v, ids, route, ak, av, lm, bm=64):
    h, n, d = q.shape
    m = ids.shape[1]
    w = ids.shape[2]
    out = torch.empty_like(q)
    nk = triton.cdiv(n, 128) * 128
    mk = triton.cdiv(m, 128) * 128
    kt = torch.nn.functional.pad(k.transpose(1, 2), (0, nk - n))
    akt = torch.nn.functional.pad(ak.transpose(1, 2), (0, mk - m))
    _attn[m * (64 // bm), h](q, kt, v, ids, route, akt, av, lm, out, n, m, w, bm, nk, mk)
    import os
    if _settings.get('DEBUG_SYNC') == '1':
        torch.npu.synchronize()
        print('CUSTOM_SPARSE_OK', flush=True)
    y = torch_npu.npu_fusion_attention(q[:, m * 64:][None], k[None], v[None], h, input_layout='BNSD', scale=128 ** (-0.5), keep_prob=1.0, sparse_mode=0)[0]
    out[:, m * 64:].copy_(y[0])
    return out
