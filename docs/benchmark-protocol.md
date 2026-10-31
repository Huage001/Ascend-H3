# Single-card end-to-end benchmark / 单卡端到端测试

Performance reporting covers **1344×768 at 124 and 345 native frames**.
The 672×384 cases remain supported but are excluded from the reported speedups.
All workloads in `configs/workloads.json` use native 124 or 345 frames,
batch one, seed 42 and the same prompt. Frame interpolation and Refine are off.
Only the selected card must be free of other compute processes. Unknown or
unverifiable processes invalidate a measurement. Never terminate other users'
processes to obtain a result.

## Reference and students

**Every method and baseline performs eight actual DiT forwards.** Speedup is
same-device, same-precision **Full + LightX2V8 end-to-end time / candidate
eight-step end-to-end time**, at the same geometry, frame count, prompt and
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

Bars show average measured DiT milliseconds per block, normalized by the
actual number of profiled forwards and 50 blocks. Headline minutes and speedups
come only from complete, uninstrumented requests. All headline results require an actual complete request at the recorded
student schedule: all methods require eight forwards, including the separate profile.

```bash
python scripts/plot_long_video.py --input benchmarks/long-video-e2e.json \
  --output docs/figures/long-video-e2e
```

The JSON and final figure are populated only after actual measurements finish.
There are no placeholder performance numbers.

## Loading memory fixes

The FastH3 V2 compatibility patch excludes only automatically generated online
INT8 gate scales from the list of tensors required in the BF16 checkpoint.
Actual trained gate weights and serialized quantization scales remain mandatory.
CUDA initialization may collect unreachable objects and trim unused host pages
before constructing each pinned CPU master. This runs only during offloader
construction, outside measured requests; it changes no tensor values, master
dtypes, residency or prefetch policy. Failed attempts stay in the raw archive.
