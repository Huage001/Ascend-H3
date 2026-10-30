# Backward performance

## Workload and measurement

The target is the `tests/benchmark_block_sparse_vs_fusion.py` default
V6A workload: B=1, H=24, S=53248 (53223 natural tokens), D=128, BF16,
BNSD, 128x128 sparse blocks, `inner_precise=0`, seed 1234. Each query
block keeps 58 of 416 KV blocks, giving 86.0577% sparsity. FA computes
unmasked dense attention, so its outputs are intentionally different.

Measurements use an otherwise idle Ascend 910B3, NPU 6, CANN 9.1.0.
The script's normal backward timing builds the forward graph before
starting the timer and synchronizes the NPU around `backward()`.
This includes backward launch, allocation, kernel execution, and final
gradient conversion. Routing/mask construction is outside the timing.

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
ASCEND_DEVICE_ID=6 python3 tests/benchmark_block_sparse_vs_fusion.py \
  --device 6 --warmup 3 --iters 10
```

## Result

**The 100 ms target was not reached.** The final clean-build default measured
**108.302 ms** backward, down from **140.436 ms** for the untouched original
binary: a **22.9% latency reduction (1.30x)**. No additional GM workspace is
required. Reaching 100 ms still requires another 8.302 ms reduction.

Final measurements use the unchanged benchmark script, three warmups and
ten timed iterations on NPU 6, with no optimization environment override:

| Stage | Original BSA | Optimized BSA | FA in optimized run | FA / optimized BSA |
| --- | ---: | ---: | ---: | ---: |
| Forward | 49.223 ms | 49.488 ms | 313.467 ms | 6.334x |
| Backward | 140.436 ms | 108.302 ms | 595.172 ms | 5.496x |

Optimized backward p50/p90 were 108.293/108.395 ms. The original binary's
backward p50/p90 were 140.519/140.643 ms. The original binary SHA256 was
`28096ce0009d8bc871773962c23b2d909b83e55c17c5651b2ba1b2e76030440f`;
the final installed library matches the clean build, SHA256
`9d46dea68cb0d02d9ca503cd5766f56315bf0259a2dead015d3d8ba6bc49e114`.

The retained changes are:

- Assign head-major work items cyclically across AICs, instead of giving
  each core a contiguous range. This improves opportunities for shared K/V
  cache reuse across cores.
- Keep two Q/dO pairs in L1 for the whole work item; score recomputation and
  dK/dV consume the same operands. Preserve eight-edge packets and all eight
  cached K slots for dQ.
- Release each score K/V reuse token after its last L1 reader, avoiding
  repeated producer fences for resident Q/dO. Full-Q dK/dV inputs use one
  repeated-fractal transpose; partial Q tiles keep the original slab loads.

L1 ownership in the optimized path:

| Region | Contents |
| --- | --- |
| 0–64 KiB | Two P/dS stream slots, shared temporally with Cube1 |
| 64–320 KiB | Eight K tiles for dQ; Cube1 also uses the first two slots |
| 320–512 KiB | Space for three Q/dO pairs; the selected schedule uses two |

Automatic dispatch now enables this path for BNSD with D=128 and either
128-token KV blocks or a 64x64 block shape, without additional KV-head-count
or Q/KV-sequence-length gates.
The original performance measurements above used the former dispatch gates
(Q/KV lengths in [8192, 65536] and at least as many KV heads as AICs); widening
dispatch does not imply the same speedup for every newly enabled shape.
Other shapes keep the legacy schedule. `BSA_BACKWARD_OPT=0` forces
the legacy path; `BSA_BACKWARD_OPT=1` forces the optimized path for eligible
BNSD/128-KV or BNSD/64x64 layouts (including short sequences and D=64 for regression).
Other environment values are rejected. All experimental tuning knobs were
removed. Forward computation and the Python API are unchanged.

## Baseline

The original implementation measured 140.041 ms backward and 48.838 ms
forward. Device profiling measured 138.058 ms for the backward kernel.
In a representative profiled call, AIC MTE2 and Fixpipe activity ratios
were 78.7% and 72.3%, versus 35.7% for matrix computation. These ratios
overlap and must not be added as a breakdown of total latency.

A later unmodified-binary recheck with 3 warmups and 10 measurements gave
138.349 ms (p50 138.404, p90 138.583). Thus changes of roughly 1-2 ms
relative to the initial baseline are not sufficient evidence of a robust
speedup across runs.

Reducing the logical Q group from 48 to 32/24/16/12/8/4 tasks did not help:
the normal backward means were 139.118/141.063/143.113/142.713/146.233/
147.448/151.759 ms respectively. These results use contiguous core ownership;
reducing Q grouping alone does not reproduce the cyclic-schedule gains.

An experiment retaining FP32 gradients in NZ order through accumulation
passed the new precision tests and exactly matched sampled V6A gradients,
but measured 141.923 ms backward. That experiment was removed.

## Other experiments

Prepacking Q/dO/K/V into NZ storage passed correctness checks but did
not improve end-to-end backward time (140.398 ms with striped packing;
153.385 ms with whole-tile packing). Its additional 1.219 GiB workspace
and code were removed. Increasing packet capacity from 8 to 12/16 also
regressed performance (147.009/180.729 ms without packing). A 32-edge
compile exceeded the AIV stack limit. The original 8-edge capacity remains;
the logical work-item size is now selected together with core ownership.

Expanding row metadata to 16 FP32 lanes before the native-NZ vector
subtractions measured 137.826 ms backward (FA: 594.152 ms). It uses the
existing UB tail, does not allocate extra GM workspace, and retains the
scale/subtract/exp and subtract/multiply arithmetic order.
This experiment was also removed because its end-to-end gain was not
distinguishable from run-to-run variation.

Cube1 Q/dO prefetch, within-KV L0B reuse, and contiguous transpose loads
passed all relevant tests and exactly matched sampled V6A gradients, but
measured 138.846 ms (10 measurements). These changes were removed.

Replacing the eight L1-to-L0 2D loads of complete 128x128 tiles with one
3D load also passed the 68 relevant tests and exactly matched sampled
V6A gradients, but measured 139.128 ms (p50 139.078, p90 139.373).
This experiment was removed as well. The retained full-Q dK/dV transpose
combines contiguous 2D fractals, not these 3D operand loads.

## Follow-up experiments (2026-09-06)

The following variants were precision-checked and then removed; their
timings do not represent the retained implementation:

| Variant | Backward mean | Observation |
| --- | ---: | --- |
| Deferred dQ, complete KV window | 135.7–137.5 ms | Only about 3 ms faster than its alternating control; adds 12.188 GiB workspace. |
| Deferred dQ, 16/32/64/128 KV windows | 139.9–143.8 ms | Smaller buffers erase the gain. |
| Deferred dQ plus dK/dV | 178.6–188.4 ms | Less overlap outweighs fewer atomic stores. |
| Four Q/dO pairs retained in L1 across both phases | 150.523 ms | 32 precision cases pass; reduced Q grouping/packet capacity outweighs operand reuse. |
| NZ gradient workspace with cyclic Q/dO residency | 110.9–111.2 ms | Precision passes, but conversion/writeback costs exceed the ordinary-output version's 106.8–107.1 ms. |

The deferred-dS experiment exposed a short-window writeback race: explicitly
draining the AIV MTE3 queue before publishing completion made its 68 related
precision cases pass. That synchronization and the deferred workspace are
not needed by the original schedule and are not retained.

Cyclic scheduling with two-Q work items but no operand residency measured
approximately 117–118 ms. Adding Q/dO residency reduced this to 107–108 ms.
The final producer-fence/transpose variant gave 107.289/107.419 ms versus
109.354/109.425 ms with those local changes disabled in the same binary.
Shrinking packets to four/two edges or adding per-core workspace padding
did not improve the selected eight-edge configuration.

Device profiling of the cyclic two-Q schedule before operand residency
measured 116.663 ms. Representative AIC MTE2 activity dropped from about
99.6 ms to 79.0 ms, while Fixpipe activity remained about 90.8 ms. These
overlapping counters indicate a changed balance; they are not an additive
latency decomposition or direct measurements of HBM transfer time.

## Correctness coverage

The regression in `tests/test_backward_wide_regression.py` compares output,
dQ, dK, and dV against masked FP32 attention. It covers both input dtypes,
D=64/128, Q blocks of 128/256, aligned and unaligned sequence lengths,
multiple batches, shared KV heads, and untouched KV columns. It explicitly
tests both legacy and optimized dispatch, including repeated cyclic work
items and one-token sequence tails.

The final clean build passed **116 cases**, with 48 KV=192/256 cases
excluded. A separate default-V6A check compared 426 sampled rows across
all 24 heads against the original binary. Output was identical; maximum
absolute dQ/dK/dV differences were 0.0001221/0.0002441/0.0004883, with
relative L2 differences 2.225e-5/2.585e-5/3.805e-5. All comparisons passed
the then-current rtol=0.02, atol=0.008 thresholds.

Run the wide regression and filtered backward accuracy suites as two commands
(the KV filter must not be
applied to the first file, whose Q=256 cases are in scope):

```bash
BSA_TEST_DEVICE=6 ASCEND_DEVICE_ID=6 python3 -m pytest -q tests/test_backward_wide_regression.py
BSA_BACKWARD_OPT=1 BSA_TEST_DEVICE=6 ASCEND_DEVICE_ID=6 python3 -m pytest -q tests/test_backward_accuracy.py -k 'not 192 and not 256'
```

The original packaged binary also times out in
`test_backward_against_dense_fp32[192-1-64-dtype1]` (KV=192, Q=1, D=64,
FP16). This was reproduced separately with a 45-second timeout. The
optimization targets KV=128; validation of the modified path uses KV=64/128
and the dedicated packet/work-item boundary regressions.

### Uniform 0.008 tolerances (2026-09-06)

All accuracy assertions in `tests/` now use `rtol=0.008, atol=0.008`.
The `close_ratio` metric uses the same bound:
`abs(actual - reference) <= 0.008 + 0.008 * abs(reference)`.
This also changes the smoke forward thresholds from 0.004 to 0.008.

Revalidation on NPU 6 passed **125 cases**: 64 wide regression cases
(legacy and optimized paths), 52 backward accuracy cases with
`BSA_BACKWARD_OPT=1`, and 9 smoke cases. The same 48 KV=192/256 backward
cases were deselected because of the known timeout described above;
this is not a full-suite pass. Smoke forward coverage includes KV=192/256.

Run the two commands above, then the smoke suite (all three files honor
`BSA_TEST_DEVICE`, defaulting to device 7):

```bash
BSA_TEST_DEVICE=6 ASCEND_DEVICE_ID=6 python3 -m pytest -q tests/test_smoke.py
```

The final device profiling report and CSV are archived in
[FINAL_PROFILE.md](FINAL_PROFILE.md).

### Removal of head-count and sequence-length dispatch gates

The D=128 optimized path is now selected automatically for BNSD/KV=128
without the former `kvHeads >= aicNum` or `[8192, 65536]` sequence gates.
The wide regression matrix now includes unset-environment automatic dispatch
in addition to forced legacy/optimized modes. Rectangular MQA tests cover
Q/KV lengths `(1,129)`, `(129,1)`, and `(129,65537)` in both
BF16 and FP16, without materializing a 65537-by-65537 reference matrix.

The retained cases have passed in targeted runs on NPU 6: 96 wide matrix
cases, 52 filtered backward accuracy cases, 9 smoke cases, and 6 rectangular
cases (163 total). The six rectangular cases were revalidated separately
(6 passed, 96 other cases deselected). The previously excluded 48 KV=192/256
backward cases were not run; this is not a full-suite pass.
All accuracy tolerances remain `rtol=atol=0.008`. Removing dispatch gates
does not establish accuracy or a performance improvement for every new shape.

## 64x64 wide backward (2026-09-08)

BNSD 64x64 now enters the wide packet pipeline by default at D=128, or
with `BSA_BACKWARD_OPT=1` at D=64/128. Host eligibility and the AIC/AIV
dispatch predicates agree on the new path. The existing runtime tile extents
handle the 64-column operands and short Q/KV tails; the 128x128 workspace
slot capacity and two-Q resident schedule remain unchanged.

`BSA_BACKWARD_OPT=0` preserves the original grouped implementation for
64x64, as does automatic D=64 dispatch. Other Q block sizes with KV=64,
TND, and KV>128 keep their existing paths.

Same-library measurements on NPU 6 use BF16 B=1/H=24/S=53248/D=128,
seed 1234, three warmups and ten timed iterations. The V6A 128x128 mask is
subdivided into 64x64 blocks without changing selected token pairs
(86.0577% sparsity); this is not freshly computed 64x64 Top-K routing.
Times are synchronized host-side backward calls, excluding forward execution,
not msprof device task timings.

| Backward path | Mean (ms) | p50 (ms) | p90 (ms) |
| --- | ---: | ---: | ---: |
| 64x64 original grouped (`BSA_BACKWARD_OPT=0`) | 204.042 | 203.221 | 205.743 |
| 64x64 wide (automatic) | 188.503 | 188.479 | 188.603 |
| 128x128 wide (automatic control) | 107.883 | 107.888 | 107.980 |
| Unmasked dense FA | 586.245 | 586.321 | 586.954 |

64x64 backward latency is reduced by 7.6% (1.082x versus the grouped path),
with a 3.110x speedup over dense FA. The 128x128 control remains near its
previous 108 ms result. The native library SHA256 for this run is
`1a0d13a65db2889acb28a120660a8a5d1fc61494184b1ea005d9abfe3d08649c`.

Regression validation passed 217 cases: 48 64x64 matrix cases, 96 existing
128-KV matrix cases, 12 rectangular cases, 52 filtered backward accuracy
cases, and 9 smoke cases. The 64x64 matrix covers BF16/FP16, D=64/128,
automatic/forced schedules, GQA, aligned and one-token-tail sequences.
Rectangular tests additionally exercise active KV tails and long KV inputs.
The same 48 KV=192/256 backward cases were not run; this is not a full-suite
pass. Accuracy tolerances remain `rtol=atol=0.008`.

The full V6A 64x64 workload also passed comparison with same-mask FP32
attention (HF32 disabled). Every output/dQ/dK/dV tensor was checked in full:
163,577,856 elements each, no nonfinite values, and 100% within tolerance.
Maximum absolute errors were 0.000443816 / 0.000958145 / 0.001193404 /
0.001075059 respectively; relative L2 errors were approximately
0.229381% / 0.236940% / 0.234572% / 0.234672%.
