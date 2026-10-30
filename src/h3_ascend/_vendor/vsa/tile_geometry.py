
import importlib as _importlib
import functools
import torch

@functools.lru_cache(maxsize=16)
def geometry(prefix_segments, grid, device, block_shape=(4, 4, 4)):
    tiles = []
    start = 0
    bt, bh, bw = block_shape
    block_size = bt * bh * bw
    for n in prefix_segments:
        for off in range(0, n, block_size):
            tiles.append(list(range(start + off, start + min(off + block_size, n))))
        start += n
    nprefix = len(tiles)
    t, h, w = grid
    for it in range(0, t, bt):
        for ih in range(0, h, bh):
            for iw in range(0, w, bw):
                tiles.append([start + (a * h + b) * w + c for a in range(it, min(it + bt, t)) for b in range(ih, min(ih + bh, h)) for c in range(iw, min(iw + bw, w))])
    sizes = torch.tensor([len(x) for x in tiles], device=device, dtype=torch.int32)
    gather = torch.tensor([x + [0] * (block_size - len(x)) for x in tiles], device=device, dtype=torch.long).flatten()
    valid = torch.arange(block_size, device=device)[None, :] < sizes[:, None]
    inv = torch.empty(start + t * h * w, dtype=torch.long)
    for i, rows in enumerate(tiles):
        inv[rows] = torch.arange(len(rows)) + i * block_size
    return (sizes, gather, valid, inv.to(device), nprefix)
