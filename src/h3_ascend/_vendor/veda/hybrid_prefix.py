
import importlib as _importlib
import triton
import triton.language as tl

@triton.jit
def _scatter(IN, INV, HEADS, OUT, S: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    row = P + tl.program_id(0) * B + tl.arange(0, B)
    lh = tl.program_id(1)
    d = tl.arange(0, D)
    head = tl.load(HEADS + lh)
    src = tl.load(INV + row, row < S, 0)
    x = tl.load(IN + (lh * N * 128 + src[:, None]) * D + d[None, :], row[:, None] < S, 0)
    tl.store(OUT + (row[:, None] * H + head) * D + d[None, :], x, row[:, None] < S)

def scatter_video(x, inv, heads, out, prefix, rows=64):
    _scatter[triton.cdiv(inv.numel() - prefix, rows), heads.numel()](x, inv, heads, out, inv.numel(), x.shape[2] // 128, out.shape[1], out.shape[2], prefix, rows)
