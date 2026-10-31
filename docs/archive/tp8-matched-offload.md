# Historical matched-Offload TP8 diagnostic

This superseded study used BF16, eight DiT updates, seed7, 1344×768 / 345 frames,
native TP8 and single-rank VAE decoding. Both DiT and text encoder streamed one
current block plus one prefetched block. It is separate from the new INT8
resident-first matrix and is not an ongoing BF16 E2E queue.

| Method | One card | Eight cards | One / eight |
| --- | ---: | ---: | ---: |
| Full + LightX2V8 | 1179.84 s | 232.85 s | 5.07× |

Eight ranks each held seven DiT attention heads. The separate attention-dispatch
profile had p95 rank max/mean 1.068; that instrumented diagnostic is not the
formal E2E timer. [Original raw record](../../benchmarks/tp8-scaling.json).
