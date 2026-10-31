# Validation status

This document records evidence, not intended support inferred from CLI flags.

| Check | Status |
| --- | --- |
| Four explicit native frame/geometry request contracts | Passed CPU tests |
| Native TP launch arguments and INT8 recipe forwarding | Passed CPU tests |
| Fourteen source CANN extensions, Torch 2.10 / CANN 9.0.1 / 910B3 | Built successfully |
| Fourteen extensions rebuilt against private CANN 9.2.0 | Built successfully; complete native smoke repeated successfully |
| Five backend classes instantiated inside pinned vLLM-Omni | Passed registry check |
| Legacy Turbo8 adapter fusion mapping | All 363 pairs / 259 native parameters match base checkpoint shapes |
| Original Turbo → public converter → earlier adapter | All 726 tensors and complete file SHA256 identical |
| Four workloads through actual native Omni request preprocessing | Passed, including 37/102 latent frames |
| Full INT8 TP2, 672×384 / 124 frames, eight complete DiT forwards | Both ranks finite; 50 blocks each step; all 50 QKV weights INT8 |
| Full INT8 TP2 complete 672×384 / 124-frame video + audio | Passed native inference, decode, exact dimensions/frame count and visual inspection; sample included |
| VSA INT8 TP4 complete video + audio, all four workloads | Passed; exact dimensions/frames; four steps per rank; 50 nonzero active gates each step; finite DiT outputs |
| Veda INT8 TP4 complete video + audio, all four workloads | Passed; exact geometry/frames, finite outputs and 50 blocks each of eight steps on all ranks; short sample visually inspected |
| BF16 Full, VSA, Spark and Veda finite outputs at 672×384/124-frame token geometry | Passed native smoke |
| Global heads 28–41 versus a local 14-head shard | Passed native smoke for BF16 Full/VSA/Spark/Veda |
| INT8 W8A8 projection output entering Full/VSA/Spark/Veda | Passed native smoke |
| VDN 128-dimensional factor versus FP64 inverse | Passed; relative L2 1.45e-7 |
| VDN dynamic graph scan with changed inputs versus eager | Passed bitwise check |
| Pinned LightX2V four-step source and prepared fusion | SHA256 verified; 312 factor pairs / 208 native targets; alpha=rank=128, shifts 6/3 |
| General partial Q-only community LoRA | CPU tensor regression passed; K/V slots unchanged |
| Audit removal before performance requests | CPU regression passed; all hooks removed and native forward restored |
| Full LightX2V4 INT8 single-card 672×384 / 124 frames | Complete video/audio passed; four finite DiT updates and all 50 blocks; visually inspected |
| VSA INT8 single-card 672×384 / 124 frames | Complete video/audio passed; four updates, 50 nonzero active gates each step |
| VDN official eight-step INT8 single-card 672×384 / 124 frames | Complete video/audio passed with all eight finite updates and 50 blocks; excluded from the four-step performance figure |
| VDN official eight-step BF16 and INT8 single-card 1344×768 / 345 frames | Both complete MP4s with audio passed; eight finite updates and all 50 blocks, full 128-dimensional state; sampled frames visually inspected |
| Qualitative gallery | Ten single-card 1344×768 / 345-frame MP4s, BF16 and INT8 columns, Full/VSA/Veda/VDN/Spark rows; same prompt and seed 42; VDN marked eight-step |
| Veda and Spark BF16 single-card 1344×768 / 345 frames | Complete four-step MP4s and independent profiling requests passed |
| Ascend high-resolution single-card four-step performance matrix | All 16 combinations passed: Full/VSA/Veda/Spark × BF16/INT8 × 124/345 frames at 1344×768; matched Offload, one formal complete request and an independent profile per row |
| PRO 6000 Full, Veda and Spark BF16, 1344×768 / 345 frames | Complete four-step MP4s and independent profiling requests passed; remaining reference combinations are in progress |
| PRO 6000 Full INT8, 1344×768 / 345 frames | Complete four-step video/audio, native Triton W8A8, all 50 INT8 QKV weights, finite outputs, formal request and independent profile passed |
| Cross-device original DiT tensors | All 535 tensors match in shape, dtype and SHA256 after lossless native repacking |
| New framework server, single-card full video matrix | In progress |
| New framework server, INT8 full video matrix | In progress |
| Four-card TP model inference for all five methods | In progress |
| Official NVIDIA VSA tile64 reference, four workloads | Completed on PRO6000; ten interleaved repetitions |
| Unified repository end-to-end performance | In progress |

Native smoke uses real workload token geometry with random QKV projection
inputs. It is not a generated-video quality test. INT8 smoke uses the native
dynamic W8A8 primitive followed by BF16 QKV; full model quantized loading and
LoRA interaction must also pass before the recipe is marked qualified.

Sharding can change dense kernel tiling and floating-point reductions. Finite
outputs and numerical error are recorded separately from bitwise equality;
neither should be inferred from the other.

For diagnostic video runs, set `H3_ASCEND_AUDIT_DIR` to a writable directory.
Each worker then records per-DiT output shapes and finiteness, all 50 block
forward counts, projection weight dtypes and nonzero VSA gate counts.
This mode synchronizes tensors and must be disabled for performance runs.

Raw native smoke records are in `benchmarks/native-smoke.json`. The dense
full-versus-shard relative L2 is at most 4.82e-4 in this check. BF16 VSA,
Spark and Veda shard outputs were bitwise identical; INT8-derived VSA differed
by relative L2 4.50e-9. These are shard comparisons, not BF16-versus-INT8
quantization-error measurements.

Gate auditing runs at the child projection before execution: ordinary layerwise
offload clears the parent block's device weights afterward. Inspecting all
weights at the end of the DiT would miscount valid gates as unloaded zeros.
CPU regressions reproduced this clear/restore behavior, checked all 50
active gates and verified complete hook removal after the audited warmup.

The complete VSA INT8 TP4 matrix is recorded in
`benchmarks/vsa-int8-tp4-matrix.json`; all four videos have audio.
The current high-resolution single-card 345-frame samples also received visual
sample inspection. Historical TP4 sample checks remain separate from the new
single-card qualitative gallery.

Frame-by-frame inspection of the VSA INT8 short sample found a brief
luminance dip near frames 82–84 (zero-based), around 3.4 seconds.
This is recorded as an observed quality limitation, independent of the
passed finite-output, gate, frame-count and geometry checks. A same-seed BF16
run reproduced the dip at the same position. Maximum adjacent downscaled
luminance changes are 18.19/255 (BF16) and 21.63/255 (INT8), both at the
84→85 transition. This descriptive metric is not a quality score. The
artifact is not exclusive to INT8; decoder versus generated-latent causality
remains unresolved. The earlier Ascend CANN, Triton and paired VSA references
also showed discontinuities around multiples of the VAE's 17-frame clips;
a fixed-latent decoder A/B is still required to assign a root cause.

The complete Veda INT8 TP4 matrix is recorded in
`benchmarks/veda-int8-tp4-matrix.json`, including all four video hashes.

The current performance baseline is same-device, same-precision **Full +
LightX2V, four steps**. Quantization gain is reported separately as BF16/INT8.
The former 50-step benchmark was cancelled at the user's request. See
[the benchmark protocol](../benchmark-protocol.md). Neither old attention-only
ratios nor eight-step Turbo8 samples are relabeled as these new four-step results.
