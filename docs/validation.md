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
A CPU regression reproduced this clear/restore behavior and checked all 50
gates across two forwards.

The complete VSA INT8 TP4 matrix is recorded in
`benchmarks/vsa-int8-tp4-matrix.json`; all four videos have audio.
Only the short 672×384 example has received visual sample inspection so far.

Frame-by-frame inspection of the VSA INT8 short sample found a brief
luminance dip near frames 82–84 (zero-based), around 3.4 seconds.
This is recorded as an observed quality limitation, independent of the
passed finite-output, gate, frame-count and geometry checks. A same-seed
BF16 run is prioritized to investigate quantization versus student/runtime
behavior; no cause is assigned from this sample alone.

The complete Veda INT8 TP4 matrix is recorded in
`benchmarks/veda-int8-tp4-matrix.json`, including all four video hashes.
