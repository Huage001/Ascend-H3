# Ascend-H3

[中文说明](README.zh-CN.md)

MiniMax H3 inference on Ascend with Full, VSA, Veda, VDN and Spark-H3
attention in one repository. The integration reuses vLLM-Omni model loading,
INT8 transformer quantization and native tensor parallel attention heads.

**Integration validation is in progress.** The source operators have compiled
successfully; this branch is not yet a qualified release of every
method × precision × workload × parallel configuration. The single-card end-to-end matrix is being remeasured against four-step Full H3.

## Workloads

| Case | Width × height | Native output frames | Latent video grid |
| --- | --- | --- | --- |
| `672-124` | 672×384 | 124 | 37×12×21 |
| `672-345` | 672×384 | 345 | 102×12×21 |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

Output FPS is 24. Requests use explicit frame counts, with frame interpolation
disabled. The long case is the agreed native 345-frame workload; it is not
cropped to 341 frames. BF16 and INT8 use the same four geometries.

## Installation

Use Linux, Python 3.12 and a compatible Ascend Torch/CANN environment.
Install the pinned vLLM, vLLM-Ascend and vLLM-Omni stack described in
[the environment notes](docs/environment.md). Do not combine unrelated
framework releases. Then install this repository from source:

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

Builds contain no model weights and run without occupying an NPU. Native
libraries are compiled locally and kept outside the checkout. For a single
backend, replace `all` with `vsa`, `veda` or `spark`; Full and VDN do not
require these custom CANN extensions. Keep the editable checkout available.

## Checkpoints and algorithm choices

| Method | Required model assets | Computation |
| --- | --- | --- |
| `full` | H3 base; LightX2V 4-step by default, or a community LoRA | Dense attention |
| `vsa` | Complete official FastH3 V2 student, including all 50 trained compression gates | 8 forwards, 80% video sparsity; logical KV64, selected KV blocks paired into KV128 on chip |
| `veda` | H3 base + LightX2V 4-step + official trained Veda predictor | Dynamic per-head Veda block routes, nominal 90% sparsity |
| `vdn` | H3 base + official VDN stage-DMD checkpoint directory | Local softmax plus 128-dimensional learned DeltaNet branch |
| `spark` | H3 base + LightX2V 4-step | Spark-H3 hierarchical partition and joint exact/proxy softmax, nominal 90% exact-branch sparsity |

Download models under their original licenses. Weights are not redistributed.
Different distilled students have different schedules and learned rules;
swapping attention on arbitrary weights does not guarantee matching quality.
VSA requires nonzero trained gates. VDN requires its learned branch weights.
Download and preparation commands are in [model assets](docs/weights.md).

Full, Veda and Spark enable the pinned **LightX2V four-step LoRA** by default.
Use `--no-few-step-lora` to run the original many-step model. A prepared
community `--lora /models/style` is fused together with LightX2V before loading
finishes; disable the default student when explicitly selecting another
community student. Veda always needs its official trained predictor.
VSA accepts only the complete official **FastH3 V2 full checkpoint**: eight
forwards, 80% video sparsity, video/audio shifts 10/3, tile64, guidance 1.
It loads the released weights directly and does not take `--lora`. VDN accepts
only its complete official stage-DMD checkpoint. Neither accepts an additional community LoRA or student-disable flag.

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full four-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

The official PEFT source is normalized once using its declared alpha/rank;
prepared files are hash-checked. All selected adapters merge before TP slicing
and INT8 quantization. There are no request-time LoRA matrix multiplications.
See [model preparation](docs/weights.md) for source formats and fixed schedules.

## Start a server

Use only devices that have no other compute processes. For one card:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision bf16 --devices 1
```

Other backends use the same server API:

```bash
h3-ascend serve --method vsa --model /models/FastH3-V2 \
  --precision bf16 --devices 1
h3-ascend serve --method veda --model /models/MiniMax-H3 \
  --predictor /models/veda/predictor.safetensors \
  --precision bf16 --devices 1
h3-ascend serve --method vdn --model /models/MiniMax-H3 \
  --lora /models/VDN/stage-dmd-step-250 --precision bf16 --devices 1
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision bf16 --devices 1
```

Run one server at a time on the selected devices. Add `--dry-run` to inspect
the generated native vLLM-Omni command without loading a model.

For **INT8**, replace `--precision bf16` with `--precision int8` for any method.
This forwards the native quantization recipe:

```json
{"transformer":{"method":"int8"}}
```

INT8 projections feed BF16 Q/K/V into attention. Attention accumulators,
Veda predictions and VDN state keep their prescribed precision. This is
INT8 model inference, not a claim that every attention operation is INT8.
The native recipe retains unsupported layers in BF16. Single-card H3 has a
96,768-wide projection that exceeds the NPU quantized matmul output limit
of 65,535; TP can reduce its per-rank width into range.

For **four-card inference**, use native tensor parallel head sharding:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp
```

`--devices` supports 1, 2, 4, 7 and 8, which divide 56 heads. Veda retains
global predictor head IDs after sharding. Seven-card DiT TP uses one text
encoder rank because its 64 query / 8 KV heads cannot be divided by seven.
All methods use **Offload: one active block plus one asynchronously prefetched
block**, for both DiT and text encoder, with pinned CPU weight masters. VAEs
remain resident. The benchmark policy is
`omni-layer-prefetch1-vae-resident-v1`; NVIDIA reference measurements must use
the same pinned Omni implementation and policy. This is an explicit memory
policy, not a different resident-layer count for each algorithm. Optional
`--parallel ulysses` is exposed for non-VDN paths; TP remains the default.

## Generate the four cases

Create a UTF-8 prompt file, start a server, then use its matching step count:

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 672-124 672-345 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 42 --output "outputs/${CASE}.mp4"
done
```

Pass the server's `--method` to the client: defaults are four steps for
Full/Veda/Spark, eight for VSA V2 and VDN. For the optional original many-step model, use
`generate --method full --no-few-step-lora` (50 steps), matching the server.
An explicit `--steps` remains available for a separately prepared community
student. `--guidance-scale` is optional; omission preserves native policy.
No Refine stage is added. The client saves the MP4 and a JSON timing receipt.

## End-to-end performance

The single-card matrix is in progress. Only the measured rows below are
qualified; missing entries remain pending. Every attention speedup uses **Full + LightX2V, four steps** on
the same hardware and at the same precision, resolution and frame count.
Quantization gain is reported separately as the same method's BF16 / INT8
end-to-end time. The earlier 50-step reference run was cancelled at the user's
request and is not used in the new comparison.

LoRA fusion, model loading and JIT warmup finish before timing. The timer
includes fresh prompt encoding, all actual denoising updates, video/audio
decoding, MP4 encoding and delivery. Full/Veda/Spark use the pinned LightX2V
four-step student; VSA now uses the complete official FastH3 V2 eight-step student.
Its ratio against Full4 is a comparison of complete configured systems with different
step counts, not an equal-step attention-only speedup. Old VSA v1 measurements
are [archived](benchmarks/archive/fasth3-v1-four-step-highres.json) and are not V2 results.
VDN requires its official eight-step configuration and is documented separately,
not included in the current measured performance matrix. These students are not claimed to produce
identical quality, and the table makes no 50-to-four-step distillation claim.

Memory-constrained cases are measured with **Offload**, not listed as OOM.
For each cross-device pair, PRO 6000 and Ascend 910B3 must use the same explicit
resident/prefetch policy, including encoder and decoder residency. The shared Offload
policy is stated below the combined comparison table. BF16 and INT8
measurements identify their actual quantized modules and fallback layers.
The 1344×768, 345-frame figure compares both devices and both precisions.
Attention/FFN breakdowns come from a separate profiling run; they never replace
measured end-to-end time. Historical operator timings remain in raw benchmark
files for provenance and are not headline video speedups.

Speedups are reported only at **1344×768, 124/345 frames**. The 672×384
workloads remain supported for inference and functional validation.

<!-- BEGIN HIGHRES RESULTS -->

1344×768 · single card · Full/Veda/Spark 4 steps, VSA V2 8 steps · seed 42 · **one formal complete request per measured device/case**:

| Frames | Precision | Method | Ascend 910B3 time | Ascend speedup | PRO 6000 time | PRO 6000 speedup |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 124 | BF16 | Full + LightX2V · 4 | 121.04 s | 1.00× | — | — |
| 124 | BF16 | VSA · FastH3 V2 · 8 · 80% | — | — | — | — |
| 124 | BF16 | Veda + LightX2V · 4 · 90% | 69.86 s | 1.73× | — | — |
| 124 | BF16 | Spark-H3 + LightX2V · 4 | 78.15 s | 1.55× | — | — |
| 124 | INT8 | Full + LightX2V · 4 | 111.90 s | 1.00× | — | — |
| 124 | INT8 | VSA · FastH3 V2 · 8 · 80% | — | — | — | — |
| 124 | INT8 | Veda + LightX2V · 4 · 90% | 61.78 s | 1.81× | — | — |
| 124 | INT8 | Spark-H3 + LightX2V · 4 | 71.41 s | 1.57× | — | — |
| 345 | BF16 | Full + LightX2V · 4 | 648.97 s | 1.00× | 250.88 s | 1.00× |
| 345 | BF16 | VSA · FastH3 V2 · 8 · 80% | — | — | — | — |
| 345 | BF16 | Veda + LightX2V · 4 · 90% | 225.29 s | **2.88×** | 103.06 s | 2.43× |
| 345 | BF16 | Spark-H3 + LightX2V · 4 | 276.45 s | **2.35×** | 115.23 s | 2.18× |
| 345 | INT8 | Full + LightX2V · 4 | 603.84 s | 1.00× | 243.86 s | 1.00× |
| 345 | INT8 | VSA · FastH3 V2 · 8 · 80% | — | — | — | — |
| 345 | INT8 | Veda + LightX2V · 4 · 90% | 203.77 s | **2.96×** | 95.43 s | 2.56× |
| 345 | INT8 | Spark-H3 + LightX2V · 4 | 247.17 s | **2.44×** | 107.27 s | 2.27× |

All records use matched **Offload** (`omni-layer-prefetch1-vae-resident-v1`). Each speedup is relative to Full four-step on the same hardware and precision. VSA V2 retains eight forwards, so its ratio compares complete configurations with different step counts. The higher speedup in each completed pair is bold; ties and incomplete pairs are unmarked. — means pending; old VSA v1 four-step numbers are not V2 results.

Fully validated: Ascend 12/16 cases; PRO 6000 6/16 cases. Remaining cases are in progress; warmups and failed attempts are excluded.

[Raw high-resolution records and independent profiles](benchmarks/single-card-highres.json).

<!-- END HIGHRES RESULTS -->

[Measurement and figure protocol](docs/benchmark-protocol.md).

## Qualitative results and validation

Ascend 910B3 only: **1344×768, 345 frames, 24 FPS (14.375 seconds, approximately 14.4 seconds)**, with audio.

Left: BF16. Right: native W8A8 INT8. All samples use one card, the same prompt, seed 42 and the same Offload policy.

Full, Veda and Spark use four steps. **VSA V2 and VDN retain their official eight steps.** VSA V2 samples are being regenerated; the old VSA v1 samples are archived, not relabeled as V2.

> A cyclist rides along a quiet green park path, smooth cinematic tracking shot, natural daylight.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Full + LightX2V · 4 steps**

https://github.com/user-attachments/assets/cd6c4619-7cef-42fc-abc2-eebea5ed4b9b

</td>
<td>

**Full + LightX2V · 4 steps**

https://github.com/user-attachments/assets/e0039d68-b50b-4f65-8714-07d4cfc3855a

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 8 steps · 80%**

New video pending validation.

</td>
<td>

**VSA · FastH3 V2 · 8 steps · 80%**

New video pending validation.

</td>
</tr>
<tr>
<td>

**Veda + LightX2V · 4 steps**

https://github.com/user-attachments/assets/c2e08051-3753-495b-bb69-7ed8b7208b2a

</td>
<td>

**Veda + LightX2V · 4 steps**

https://github.com/user-attachments/assets/ef130119-c58f-4cff-86c2-706524bbb550

</td>
</tr>
<tr>
<td>

**VDN · official student · 8 steps**

https://github.com/user-attachments/assets/bc743b7a-8587-46ea-9707-a93e514befa8

</td>
<td>

**VDN · official student · 8 steps**

https://github.com/user-attachments/assets/6144e35d-1a10-4d6c-91d0-55f469bae653

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V · 4 steps**

https://github.com/user-attachments/assets/6ebfcaed-b023-42d5-b850-e86897c0209d

</td>
<td>

**Spark-H3 + LightX2V · 4 steps**

https://github.com/user-attachments/assets/aba1d8ee-c178-4cd2-bb63-5e4a4c9f979e

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 uses **80% video sparsity** (20% keep budget); Veda retains **90%** (10% keep budget); text/audio prefixes retain the native dense policy, with block-rounding differences in the effective fraction.

[Validated videos and pending V2 samples](examples/videos/qualitative-1344-345.json). Completed videos passed dimensions, 345-frame, audio and finite-output checks. [Old VSA v1 gallery provenance](examples/videos/qualitative-1344-345-fasth3-v1-archive.json).

The same prompt and seed do not establish pixel or perceptual equivalence across students. VDN uses a full 128-dimensional numerical inverse approximation; temporal-convolution rounding is described in [architecture](docs/architecture.md). Earlier VSA samples exhibited brightness dips, documented in [temporal diagnostics](benchmarks/vsa-temporal-sample-check.json); no flicker-free claim is made.

The initial integration targets batch-one T2VA without reference videos.
See [architecture](docs/architecture.md), [validation status](docs/validation.md)
and [third-party notices](THIRD_PARTY_NOTICES.md).
