Historical matched-Offload results from 2026-10-31. These records remain separate from the resident-first matrix; explicitly reused cells retain their original provenance.

## Historical matched-Offload single-card E2E reference

**All methods, the baseline and timed requests use eight actual DiT forwards.** Full/Veda/Spark use the pinned official LightX2V 8-step 768p LoRA (shifts 6/3); VSA uses official FastH3 V2 (80% sparsity, shifts 10/3); VDN retains its official eight-step student. Veda remains at 90% sparsity. Each student keeps its trained schedule; four-step weights are not stretched to eight steps.

Speedup is **BF16 Full eight-step end-to-end time / candidate eight-step end-to-end time**, at the same hardware, geometry, frame count and seed. Both BF16 and INT8 candidates use this BF16 Full baseline, including Full INT8. Reporting covers single-card 1344×768 at 124/345 frames, BF16 and INT8. LoRAs merge before TP slicing and quantization. Timing includes prompt encoding, all eight updates, audio/video decoding, MP4 encoding and complete delivery; loading, fusion, warmup and visual screening are excluded.

At the user's request, VSA seed screening selected common seed **7**. Every method uses the same prompt and seed for formal timing, with the screening disclosed. The table accepts only newly measured results; old four-step measurements are not doubled or converted. [Earlier Full4-reference results](../../benchmarks/archive/pre-unified8-full4-reference.json) are historical only.

<!-- BEGIN HIGHRES RESULTS -->

All methods: eight forwards; common seed: 7

| Frames | Precision | Method | Ascend 910B3 time | Ascend speedup | PRO 6000 time | PRO 6000 speedup |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 124 | BF16 | Full + LightX2V8 | 206.71 s | 1.00× | 91.20 s | 1.00× |
| 124 | BF16 | VSA · FastH3 V2 · 80% | 163.20 s | **1.27×** | 79.20 s | 1.15× |
| 124 | BF16 | Veda + LightX2V8 · 90% | 113.43 s | **1.82×** | 53.60 s | 1.70× |
| 124 | BF16 | VDN · official | 301.42 s | 0.69× | 102.39 s | **0.89×** |
| 124 | BF16 | Spark-H3 + LightX2V8 | 132.78 s | **1.56×** | 61.81 s | 1.48× |
| 124 | INT8 | Full + LightX2V8 | 189.57 s | **1.09×** | 85.06 s | 1.07× |
| 124 | INT8 | VSA · FastH3 V2 · 80% | 146.89 s | **1.41×** | 74.92 s | 1.22× |
| 124 | INT8 | Veda + LightX2V8 · 90% | 97.61 s | **2.12×** | 48.28 s | 1.89× |
| 124 | INT8 | VDN · official | 213.95 s | **0.97×** | 97.74 s | 0.93× |
| 124 | INT8 | Spark-H3 + LightX2V8 | 116.19 s | **1.78×** | 54.38 s | 1.68× |
| 345 | BF16 | Full + LightX2V8 | 1179.84 s | 1.00× | 467.81 s | 1.00× |
| 345 | BF16 | VSA · FastH3 V2 · 80% | 661.43 s | **1.78×** | 318.03 s | 1.47× |
| 345 | BF16 | Veda + LightX2V8 · 90% | 376.99 s | **3.13×** | 173.62 s | 2.69× |
| 345 | BF16 | VDN · official | 842.14 s | 1.40× | 299.42 s | **1.56×** |
| 345 | BF16 | Spark-H3 + LightX2V8 | 472.93 s | **2.49×** | 196.73 s | 2.38× |
| 345 | INT8 | Full + LightX2V8 | 1125.43 s | **1.05×** | 454.32 s | 1.03× |
| 345 | INT8 | VSA · FastH3 V2 · 80% | 617.55 s | **1.91×** | 304.01 s | 1.54× |
| 345 | INT8 | Veda + LightX2V8 · 90% | 338.55 s | **3.48×** | 157.90 s | 2.96× |
| 345 | INT8 | VDN · official | 620.67 s | **1.90×** | 281.81 s | 1.66× |
| 345 | INT8 | Spark-H3 + LightX2V8 | 419.44 s | **2.81×** | 182.20 s | 2.57× |

Matched **Offload**: `omni-layer-prefetch1-vae-resident-v1`. Both BF16 and INT8 ratios use same-hardware BF16 Full8; values below 1× are slower than that baseline. The higher speedup in each completed hardware pair is bold; ties and missing values are not. — means pending; old four-step or old-seed data cannot enter this table.

Validated: Ascend 20/20, PRO 6000 20/20.

[Raw eight-step records](../../benchmarks/single-card-highres.json)

<!-- END HIGHRES RESULTS -->

These are complete-request speedups with eight updates on both sides, not
50-to-8-step distillation gains or attention-kernel benchmarks. Attention
sparsity does not reduce FFN, encoder, decoder or MP4 work; the matched Offload
policy also remains active. INT8 quantizes supported transformer linear layers,
not every attention operation. A larger relative speedup on Ascend does not
imply a lower absolute latency than PRO 6000: compare the time columns too.
VSA V2 uses 80% sparsity, whereas Veda and Spark use 90%, so those rows also
compare different attention mechanisms and students rather than equal-quality
or equal-sparsity kernels.

The earlier E2E module-profile figure is retained as [historical diagnostic data](../../benchmarks/long-video-e2e.json); its broad module spans are excluded from the attention-core comparison above.

[Earlier BF16 TP8 diagnostic with matched layer Offload](tp8-matched-offload.md), excluded from the new resident-first INT8 matrix.
