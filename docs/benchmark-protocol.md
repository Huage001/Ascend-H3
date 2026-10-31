# Benchmark protocol / 测试口径

The next end-to-end campaign measures **INT8 only** on Ascend 1/2/4/8 cards.
The Ascend/PRO 6000 **attention-core-only** and complete DiT-block comparison uses
**BF16 on both devices**: layers 0, 24 and 49, each run three times after
untimed warmup. Sum the nine timings and divide by nine for the per-layer
latency. Layer weights and inputs remain resident during these measurements;
VDN attention includes both its softmax and linear cores. These are layer
measurements, not BF16 end-to-end runs.

## Current attention timer boundary

Prepare Q/K/V, Q/K normalization and RoPE before starting the timer. Stop when
the attention core produces its output, before the final output projections.
Exclude QKV projection, Q/K norm, RoPE, output projection, AdaLN and FFN.
Include method-required sparse gates, prediction, block selection, token
rearrangement/restoration and, for VDN, both softmax and linear core branches.
Do not supply an already computed sparse plan to one method while charging
online selection to another. All required FP32 predictors/state arithmetic
remain included in the BF16 protocol.

Measure layers 0, 24 and 49, two untimed warmups and three formal runs each,
using device synchronization immediately before and after the core call.
Report the sum of all nine formal timings divided by nine. Use identical
serialized synthetic inputs on both platforms and record their tensor hashes.
These inputs are not activations captured from an actual video request.
Core and complete-block scopes are measured independently. The complete block
includes all projections, normalization, RoPE, core, AdaLN, FFN and residuals;
its latency is reported separately. Previous whole-attention-module intervals
are diagnostic raw data only and must not populate an attention-core table.

Only accepted core records may populate `benchmarks/bf16-three-layer.json`'s
`core_ms` fields. A missing core result remains pending; never substitute a
module measurement or estimate it by subtraction.

## Current INT8 end-to-end protocol

Start with resident execution (`--offload none`); preserve any OOM evidence and
try text-encoder-only streaming (`--offload text`), then whole-component swapping
(`--offload module`, qualification pending), before streaming the DiT
(`--offload layer`). Mark the actual policy for every accepted result. These
new measurements remain separate from the completed matched-Offload reference
documented below. BF16 end-to-end inference remains supported but is not rerun.

The remaining matched-Offload sections describe the earlier completed E2E
reference. Their broad module profiling spans are not the current attention
core benchmark and must not be reported as attention-core latency.

The current dense core references are Ascend's
`torch_npu.npu_fused_infer_attention_score` in BNSD layout and the pinned
official Miowtion FA4 provider on PRO 6000. Historical CUDA cuDNN baselines and
earlier layer-median aggregation are separate experiments; their ratios must
not be substituted into this nine-run comparison.

Performance reporting covers **1344×768 at 124 and 345 native frames**.
All workloads in `configs/workloads.json` use native 124 or 345 frames,
batch one, reviewed common seed 7 and the same prompt. Frame interpolation and Refine are off.
Only the selected card must be free of other compute processes. Unknown or
unverifiable processes invalidate a measurement. Never terminate other users'
processes to obtain a result.

Hardware labels identify the accelerator model. On the Ascend test host,
345-frame BF16 groups use physical card 1, 345-frame INT8 groups use card 3,
and 124-frame groups use card 3 except Spark BF16, which uses card 1 after
the long-video retake finishes. At most two suite jobs run at once.
The PRO 6000 reference uses its single GPU. The selected cards are guarded;
the protocol does not claim isolation of the entire host CPU or memory bus.

## Reference and students

**Every method and baseline performs eight actual DiT forwards.** Speedup is
same-hardware **BF16 Full + LightX2V8 end-to-end time / candidate
eight-step end-to-end time**, for both BF16 and INT8 candidates, at the same geometry, frame count, prompt and
selected common seed. No 50/8 or 4/8 multiplier is applied. Earlier Full4 and
seed42 results are archived and cannot enter this matrix.

| Method | LoRA / learned assets | Actual forwards | Video / audio shifts |
| --- | --- | ---: | --- |
| Full baseline | Pinned official LightX2V 8-step v1.0 768p | 8 | 6 / 3 |
| VSA V2 | Complete official FastH3 V2 with 50 trained gates | 8 | 10 / 3 |
| Veda | LightX2V8 + official trained predictor | 8 | 6 / 3 |
| VDN | Official stage-DMD-step-250 student, full 128-dimensional state | 8 | 12 / 3 |
| Spark-H3 | LightX2V8 | 8 | 6 / 3 |

Keep each student's trained sigma positions and shifts. Do not stretch a
four-step checkpoint into an eight-step run. VSA V2 uses 80% nominal video
sparsity; Veda and Spark's exact branch retain their 90% defaults. Native
text/audio and prefix rules remain unchanged; block rounding can change the
effective fraction.

At the user's request, VSA long videos were visually screened with new seeds;
common seed **7** was chosen. [Selection evidence](../benchmarks/seed-selection-unified8.json). Store every candidate and the selection
record. Screening is excluded from timing. After selection, all methods and
both hardware platforms use the same prompt and seed for every warmup,
formal request and profile. Different students are not asserted to have
identical perceptual quality. Do not choose the fastest timing sample.

## Matched Offload

Policy ID: `omni-layer-prefetch1-vae-resident-v1`.

- Use the same pinned vLLM-Omni `LayerWiseOffloadBackend` on both devices.
- Stream the DiT and text encoder from pinned CPU weight masters. Keep one
  current block and asynchronously prefetch one next block on a separate stream.
- Release a block's device weight storage after its forward. The circular ring
  prefetches block zero after the last block.
- Keep non-block DiT state and both VAEs resident, with the same decoder tiling
  and precision policy. Do not silently select a different resident-layer count
  or component placement for Full, sparse, BF16, INT8 or NVIDIA.
- Record **Offload** in every table and figure using this policy. A memory
  failure triggers a measured smaller-residency/chunking retry on **both**
  devices; it is not a final OOM result. Preserve failed attempts and identify
  the new common policy rather than merging incompatible measurements.

INT8 uses each platform's native Omni online W8A8 transformer recipe. Attention
Q/K/V remain BF16; FP32 predictor/state arithmetic stays prescribed. Record
fallback layers, including Ascend's output-width limit of 65,535. INT8 does not
mean every operation, encoder or VAE is INT8.

## Timing

Finish checkpoint loading, all static LoRA merges, INT8 weight conversion and
JIT warmup before timing. Run an audited warmup to verify finite DiT outputs,
all 50 blocks, actual weight dtypes and all 50 VSA gates. Remove the audit hooks
and restore the original forward before a further untimed warmup. Every student retains all eight actual forwards during every warmup.

Measure from fresh prompt submission to complete MP4 receipt and file write.
Include text encoding, all denoising updates, audio/video decode, MP4 encoding,
and response transfer. Do not cache prompt embeddings between measured requests.
Store individual timings and the number of repetitions; use their median,
without inventing error bars for a single sample. Check actual MP4 dimensions,
frame count, audio track and process ownership before accepting the result.

## Figure and profiling

`scripts/plot_long_video.py` renders PNG, SVG and PDF from a complete measured
1344×768 / 345-frame matrix. It rejects missing methods, invalid samples,
contaminated runs and mismatched Offload policies. Its input records carry:

```text
matrix_id="unified8-v1", benchmark_seed, student_profile,
hardware, precision, method, baseline, actual_steps=8,
validated, contention_detected, offload, offload_policy,
e2e_seconds: [individual uninstrumented request times],
profile: {actual_forwards, dit_wall_seconds, attention_seconds, ffn_seconds}
```

Use a separate instrumented request for Attention/FFN spans. Device-event
intervals cover non-overlapping top-level attention and FFN calls, including
QKV/output projection within the attention module. The remaining **DiT** wall
time includes unhidden transfers, modulation and dispatch gaps; do not add all
copy-stream time to compute time when they overlap.
These are observed module intervals, not isolated kernel compute times:
dependency waits and resource contention can appear inside a module span too.

Bars show average measured DiT milliseconds per block, normalized by the
actual number of profiled forwards and 50 blocks. Headline minutes and speedups
come only from complete, uninstrumented requests. All headline results require an actual complete request at the recorded
student schedule: all methods require eight forwards, including the separate profile.

```bash
python -m pip install -e '.[plotting]'  # Figure generation can run on a CPU-only host.
python scripts/plot_long_video.py --input benchmarks/long-video-e2e.json \
  --output docs/figures/long-video-e2e
```

The JSON and final figure are populated only after actual measurements finish.
There are no placeholder performance numbers.

## Interpreting the speedups

Complete-request ratios include encoder, FFN, projection, decoder and MP4 work
that sparse attention does not remove. All methods and the Full baseline use
eight updates, so no step-reduction benefit appears in these ratios. INT8
targets supported linear layers rather than the entire attention computation.
Compare absolute latency alongside relative speedup: a faster Full baseline
can produce a smaller ratio even when sparse inference finishes sooner.

The optional `attention_dispatch` event brackets the `block.attn.attention`
child, while `attention_seconds` brackets the entire `block.attn` module.
Method-specific preparation or another branch can run outside that child.
Neither event is a universally complete attention-kernel benchmark. In
particular, do not quote a dispatch-only VDN ratio as acceleration of its
whole hybrid attention module. The independent profile and the uninstrumented
formal request are different runs; their times must not be subtracted to
claim an exact transfer-time or fixed-overhead decomposition. Such attribution
requires a separate timeline trace.

## Loading memory fixes

The FastH3 V2 compatibility patch excludes only automatically generated online
INT8 gate scales from the list of tensors required in the BF16 checkpoint.
Actual trained gate weights and serialized quantization scales remain mandatory.
CUDA initialization may collect unreachable objects and trim unused host pages
before constructing each pinned CPU master. This runs only during offloader
construction, outside measured requests; it changes no tensor values, master
dtypes, residency or prefetch policy. Failed attempts stay in the raw archive.

For VSA and VDN BF16, both platforms use exact file-backed masters for the 27 input-only
vision blocks. These blocks do not execute in T2VA; active text and DiT masters
retain the pinned one-current/one-prefetch policy. This matched initialization
setting is recorded separately as `dormant_input_vision_file_backed`; older
BF16 records for these methods without it are superseded by a complete retake.

The NPU VDN local branch bounds temporary KV gathers by batching complete
independent windows. It retains each window's keys and normalization rather
than changing model weight residency. This is an NPU workspace allocation
change; the NVIDIA kernel implementation remains its platform reference.
No native bitwise equivalence is asserted for the changed CANN call shapes.
Independent profiling also releases unreachable host response objects after
all formal timing has finished, before installing profiling hooks.
