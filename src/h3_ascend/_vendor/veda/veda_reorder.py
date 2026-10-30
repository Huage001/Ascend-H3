
"""Layer/head-specific ragged packed arena; unchanged Veda groups and arithmetic."""
import importlib as _importlib
import torch
import triton
import triton.language as tl

@triton.jit
def _gather_all(Q, K, V, G, S, HEAD, GO, SO, NN, OFF, QT, KT, VT, QS: tl.constexpr, QH: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr):
    tile = tl.program_id(0)
    lh = tl.program_id(1)
    kind = tl.program_id(2)
    n = tl.load(NN + lh)
    head = tl.load(HEAD + lh)
    g = tl.load(GO + lh)
    s = tl.load(SO + lh)
    o = tl.load(OFF + lh)
    count = tl.load(S + s + tile, tile < n, 0)
    r = tl.arange(0, 128)
    d = tl.arange(0, 128)
    src = tl.load(G + g + tile * 128 + r, tile < n, 0)
    valid = (tile < n) & (r[:, None] < count)
    if kind == 0:
        x = tl.load(Q + src[:, None] * QS + head * QH + d[None, :], valid, 0)
    elif kind == 1:
        x = tl.load(K + src[:, None] * KS + head * KH + d[None, :], valid, 0)
    else:
        x = tl.load(V + src[:, None] * VS + head * VH + d[None, :], valid, 0)
    pos = o + (tile * 128 + r[:, None]) * 128 + d[None, :]
    if kind == 0:
        tl.store(QT + pos, x, tile < n)
    elif kind == 1:
        tl.store(KT + pos, x, tile < n)
    else:
        tl.store(VT + pos, x, tile < n)

@triton.jit
def _pool_all(QT, KT, S, SO, NN, OFF, FO, FQ, FK):
    tile = tl.program_id(0)
    lh = tl.program_id(1)
    kind = tl.program_id(2) // 2
    n = tl.load(NN + lh)
    s = tl.load(SO + lh)
    o = tl.load(OFF + lh)
    f = tl.load(FO + lh)
    r = tl.arange(0, 128)
    d = tl.program_id(2) % 2 * 64 + tl.arange(0, 64)
    count = tl.load(S + s + tile, tile < n, 0)
    pos = o + (tile * 128 + r[:, None]) * 128 + d[None, :]
    if kind == 0:
        x = tl.load(QT + pos, tile < n, 0).to(tl.float32)
    else:
        x = tl.load(KT + pos, tile < n, 0).to(tl.float32)
    valid = r[:, None] < count
    mean = tl.sum(x, 0) / tl.maximum(count, 1)
    mx = tl.max(tl.where(valid, x, float('-inf')), 0)
    mn = tl.min(tl.where(valid, x, float('inf')), 0)
    dest = f + tile * 384 + d
    if kind == 0:
        tl.store(FQ + dest, tl.where(count > 0, mean, 0), tile < n)
        tl.store(FQ + dest + 128, tl.where(count > 0, mx, 0), tile < n)
        tl.store(FQ + dest + 256, tl.where(count > 0, mn, 0), tile < n)
    else:
        tl.store(FK + dest, tl.where(count > 0, mean, 0), tile < n)
        tl.store(FK + dest + 128, tl.where(count > 0, mx, 0), tile < n)
        tl.store(FK + dest + 256, tl.where(count > 0, mn, 0), tile < n)

class LayerPack:

    def __init__(self, m, layer, reuse=False):
        self.reuse = reuse
        self.model = m
        self.layer = layer
        self.groups = m.groups[layer]
        self.keyptr = {}
        self.views = {}
        self.meta = {}
        go = []
        so = []
        nn = []
        off = []
        fo = []
        heads = []
        gr = []
        ct = []
        gstarts = {}
        cstarts = {}
        p = 0
        f = 0
        for key, hs in self.groups:
            layout = m.layouts[key]
            n = layout.n_tiles
            h = hs.numel()
            if key not in gstarts:
                gstarts[key] = sum((t.numel() for t in gr))
                cstarts[key] = sum((t.numel() for t in ct))
                gr.append(layout.gather_index)
                ct.append(layout.valid_count)
            self.meta[id(layout), hs.data_ptr()] = (p, f, h, n)
            for head in hs.cpu().tolist():
                heads.append(head)
                go.append(gstarts[key])
                so.append(cstarts[key])
                nn.append(n)
                off.append(p)
                fo.append(f)
                p += n * 128 * 128
                f += n * 384
        self.maxn = max(nn)
        self.elements = p
        self.felements = f
        self.g = torch.cat(gr)
        self.s = torch.cat(ct)
        self.head, self.go, self.so, self.nn, self.off, self.fo = [torch.tensor(x, device=m.device, dtype=torch.int64) for x in (heads, go, so, nn, off, fo)]
        assert len(heads) == m.num_heads and len(set(heads)) == m.num_heads

    def allocate(self, dtype, device):
        if self.reuse and hasattr(self, 'tiles'):
            return
        self.tiles = [torch.empty(self.elements, dtype=dtype, device=device) for _ in range(3)]
        self.features = [torch.empty(self.felements, dtype=torch.float32, device=device) for _ in range(2)]
        self.views = {key: tuple([x[p:p + h * n * 128 * 128].view(1, h, n * 128, 128) for x in self.tiles] + [x[f:f + h * n * 384].view(h, n, 384) for x in self.features]) for key, (p, f, h, n) in self.meta.items()}

    def reorder(self, xs):
        self.allocate(xs[0].dtype, xs[0].device)
        _gather_all[self.maxn, self.head.numel(), 3](*xs, self.g, self.s, self.head, self.go, self.so, self.nn, self.off, *self.tiles, *xs[0].stride()[:2], *xs[1].stride()[:2], *xs[2].stride()[:2])

    def pool(self):
        _pool_all[self.maxn, self.head.numel(), 4](self.tiles[0], self.tiles[1], self.s, self.so, self.nn, self.off, self.fo, *self.features)

    def prepare(self, xs):
        self.reorder(xs)
        self.pool()

    def pack(self, q, k, v, layout, heads, channels, mode):
        return self.views[id(layout), heads.data_ptr()]
