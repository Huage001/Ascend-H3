# Integration design

The vLLM-Omni pipeline owns tokenization, text/audio/video encoders, denoising,
LoRA loading, quantized projections, head parallelism and media decoding.
Ascend-H3 replaces attention computation through an explicit general plugin.
Without `H3_ASCEND_METHOD`, importing the package changes no backend.

## Attention paths

* **Full:** CANN fused inference attention. Explicit masks in auxiliary token
  refiners use the upstream masked SDPA path.
* **VSA:** spatial-temporal logical blocks of 4×4×4 tokens, pooled routing,
  selected-token softmax and trained compressed attention gates. The CANN
  kernel pairs selected KV64 blocks into KV128 compute tiles. Invalid edge
  tokens remain masked. Text/audio prefix regions retain dense attention.
* **Veda:** pretrained per-layer/per-head route prediction, per-head token
  permutations, fused route/pack stages, CANN sparse softmax and inverse
  permutation. Native TP slices data heads, while predictor calls retain the
  original global head IDs. Selection is recomputed for every attention call.
* **Spark:** content-dependent metric, hierarchical balanced token partition
  and pooled route selection. Selected blocks contribute their original K/V
  tokens. Each unselected block contributes an anchor-conditioned K/V proxy
  with a log-mass correction. One online softmax jointly normalizes those
  contributions and the dense prefix. This preserves the upstream Spark
  reweight mechanism. CANN stable partitions, terminal routing and a shared
  captured tree reduce dispatch overhead. Workspaces and launch plans are
  reusable; content-dependent routes are not.
* **VDN:** upstream learned local-softmax plus full-width DeltaNet linear
  branch. Complete frame windows are grouped to bound each temporary KV gather to 1 GiB; a softmax sequence is never split across calls. A centered Newton
  inverse and captured bidirectional scans replace expensive host launches.
  The default retains all 128 state coordinates. Newton inversion is
  numerically approximate and is not advertised as bitwise Cholesky.
  The automatic temporal-convolution policy also uses a BF16 Conv2D output
  for most geometries before FP32 activation/normalization, introducing an
  earlier rounding boundary than the upstream FP32 temporal-tap loop.
  The short 672×384 case keeps the FP32 tap calculation in a compiled graph.

## Quantization boundary

`--precision int8` selects the native vLLM-Omni transformer INT8 recipe.
Projection results enter attention as BF16 tensors. FP32 predictor, routing,
softmax accumulators and VDN state operations keep their defined precision.
Sparse attention is not itself advertised as INT8 matrix arithmetic.

## Parallelism and lifetime

Default multi-card launches use vLLM tensor parallel head sharding: QKV and
output projections are managed upstream, and this package sees local heads.
56 heads support TP sizes 1, 2, 4, 7 and 8. Veda translates local IDs back to
global checkpoint head IDs. Optional Ulysses is exposed separately; upstream
VDN currently forbids Ulysses and distributed layerwise offload. All comparisons
use ordinary layerwise offload. VSA loads the complete official FastH3 V2
checkpoint in its released Diffusers layout, including 50 trained gates, with
80% video sparsity, tile64, eight forwards and shifts 10/3. No additional LoRA
is fused into V2. The earlier v1 adapter-fusion patch remains a dependency
compatibility patch; v1 measurements are archived, not V2 results.

The initial release targets one T2VA document per request. Plan objects hold
mutable scratch tensors and NPU graph buffers, so one worker must serialize
attention execution on its stream. The default LRU holds one shape/stream
plan; VDN scan and window caches are process-local. Independent servers must
use separate device allocations. Ref-video, I2V, multi-document co-packing
and concurrent calls sharing the same buffers require additional validation.

## Build and distribution

Only source is distributed. CANN operators compile into an external cache
keyed by source hash, Torch/NPU ABI, SoC and CANN installation. Build receipts
record source and library hashes. Use an editable source installation so
`configs/` and `csrc/` remain available. A wheel-only installation is not
currently supported.

For VDN, `H3_ASCEND_VDN_TEMPORAL=ge` selects the FP32 temporal-tap graph
for every geometry; `auto` is the optimized default described above. This
option does not replace Newton inversion with Cholesky. The historical VDN
ratios use the automatic policy and should not be presented as a bitwise
implementation of the upstream feature and inverse computations.

## Long-video memory handling

VDN groups complete packed windows rather than materializing every repeated KV
row at once. Selected tokens, per-window normalization and masks are unchanged.
The CANN call shape changes, so this path is not advertised as bitwise identical
to the earlier all-window batch. Same-seed decoded video comparisons are recorded
separately from operator correctness and from formal latency measurements.

VSA and VDN BF16 initialize the 27 input-vision blocks with exact file-backed CPU
masters. For T2VA those blocks do not execute; active text and DiT weights keep
the same pinned current-block/one-prefetch policy. No weight precision is reduced.
The mapping retains all values, shapes and strides, so input-vision weights can
still be paged in, but conditioned-input performance is not part of this release's
validation. Both comparison platforms use this initialization setting for VSA and VDN
BF16. Set `H3_ASCEND_DORMANT_VISION_MMAP=0` only to investigate the earlier host
memory layout; that setting is outside the reported matched configuration.

Independent profiling runs perform host garbage collection after formal timing
has finished, before profiling hooks are installed. This cleanup is not inserted
into a timed denoising forward.
