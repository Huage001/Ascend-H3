# Single-card end-to-end benchmark / 单卡端到端测试

All four workloads in `configs/workloads.json` use native 124 or 345 frames,
batch one, seed 42 and the same prompt. Frame interpolation and Refine are off.
Only the selected card must be free of other compute processes. Unknown or
unverifiable processes invalidate a measurement. Never terminate other users'
processes to obtain a result.

## Reference and students

The attention speedup is **same-device, same-precision Full + LightX2V,
four steps / candidate end-to-end time**. Quantization gain is the same
method's BF16 / INT8 end-to-end time. All four-step candidates run the same
resolution and native frame count. No 50/4 multiplier is applied. The earlier
50-step baseline was cancelled at the user's request.

| Row | LoRA / learned assets | Actual updates |
| --- | --- | ---: |
| Full (baseline) | Pinned LightX2V v1.0 768p | 4 |
| VSA | Complete official FastH3 VSA adapter | 4 |
| Veda | LightX2V + official trained predictor | 4 |
| Spark-H3 | LightX2V | 4 |

VDN retains its official eight-step student in the inference implementation.
It is documented separately and excluded from the four-step candidate figure;
forcing it to four steps would violate its released configuration.

Different students and attention mechanisms can produce different videos.
A speedup does not assert perceptual equivalence. Keep the generated MP4s,
exact checkpoint identities, schedules and known quality limitations.

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
and restore the original forward before a further untimed warmup. All compared students retain their fixed four-step schedule during warmup.

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
hardware, precision, method, baseline, actual_steps,
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
come only from complete, uninstrumented requests. All headline results require an actual complete four-step request.

```bash
python scripts/plot_long_video.py --input benchmarks/long-video-e2e.json \
  --output docs/figures/long-video-e2e
```

The JSON and final figure are populated only after actual measurements finish.
There are no placeholder performance numbers.
