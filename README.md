# Ascend-H3

[中文说明](README.zh-CN.md)

MiniMax H3 inference on Ascend with Full, VSA, Veda, VDN and Spark-H3
attention in one repository. The integration reuses vLLM-Omni model loading,
INT8 transformer quantization and native tensor parallel attention heads.

**Integration validation is in progress.** The source operators have compiled
successfully; this branch is not yet a qualified release of every
method × precision × workload × parallel configuration. The single-card end-to-end matrix is being remeasured against eight-step Full H3.

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
| `full` | H3 base; LightX2V 8-step by default, or a community LoRA | Dense attention |
| `vsa` | Complete official FastH3 V2 student, including all 50 trained compression gates | 8 forwards, 80% video sparsity; logical KV64, selected KV blocks paired into KV128 on chip |
| `veda` | H3 base + LightX2V 8-step + official trained Veda predictor | Dynamic per-head Veda block routes, nominal 90% sparsity |
| `vdn` | H3 base + official VDN stage-DMD checkpoint directory | Local softmax plus 128-dimensional learned DeltaNet branch |
| `spark` | H3 base + LightX2V 8-step | Spark-H3 hierarchical partition and joint exact/proxy softmax, nominal 90% exact-branch sparsity |

Download models under their original licenses. Weights are not redistributed.
Different distilled students have different schedules and learned rules;
swapping attention on arbitrary weights does not guarantee matching quality.
VSA requires nonzero trained gates. VDN requires its learned branch weights.
Download and preparation commands are in [model assets](docs/weights.md).

Full, Veda and Spark enable the pinned **LightX2V eight-step LoRA** by default.
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
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
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

Pass the server's `--method` to the client: all five methods
(Full/VSA/Veda/VDN/Spark) default to eight actual forwards. For the optional original many-step model, use
`generate --method full --no-few-step-lora` (50 steps), matching the server.
An explicit `--steps` remains available for a separately prepared community
student. `--guidance-scale` is optional; omission preserves native policy.
No Refine stage is added. The client saves the MP4 and a JSON timing receipt.

## End-to-end performance

**All methods, the baseline and timed requests use eight actual DiT forwards.** Full/Veda/Spark use the pinned official LightX2V 8-step 768p LoRA (shifts 6/3); VSA uses official FastH3 V2 (80% sparsity, shifts 10/3); VDN retains its official eight-step student. Veda remains at 90% sparsity. Each student keeps its trained schedule; four-step weights are not stretched to eight steps.

Speedup is **Full eight-step end-to-end time / candidate eight-step end-to-end time**, at the same hardware, precision, geometry, frame count and seed. Reporting covers single-card 1344×768 at 124/345 frames, BF16 and INT8. LoRAs merge before TP slicing and quantization. Timing includes prompt encoding, all eight updates, audio/video decoding, MP4 encoding and complete delivery; loading, fusion, warmup and visual screening are excluded.

At the user's request, VSA seed screening selected common seed **7**. Every method uses the same prompt and seed for formal timing, with the screening disclosed. The table accepts only newly measured results; old four-step measurements are not doubled or converted. [Earlier Full4-reference results](benchmarks/archive/pre-unified8-full4-reference.json) are historical only.

<!-- BEGIN HIGHRES RESULTS -->

All methods: eight forwards; common seed: 7

| Frames | Precision | Method | Ascend 910B3 time | Ascend speedup | PRO 6000 time | PRO 6000 speedup |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 124 | BF16 | Full + LightX2V8 | — | — | — | — |
| 124 | BF16 | VSA · FastH3 V2 · 80% | — | — | — | — |
| 124 | BF16 | Veda + LightX2V8 · 90% | — | — | — | — |
| 124 | BF16 | VDN · official | — | — | — | — |
| 124 | BF16 | Spark-H3 + LightX2V8 | — | — | — | — |
| 124 | INT8 | Full + LightX2V8 | — | — | — | — |
| 124 | INT8 | VSA · FastH3 V2 · 80% | — | — | — | — |
| 124 | INT8 | Veda + LightX2V8 · 90% | — | — | — | — |
| 124 | INT8 | VDN · official | — | — | — | — |
| 124 | INT8 | Spark-H3 + LightX2V8 | — | — | — | — |
| 345 | BF16 | Full + LightX2V8 | — | — | — | — |
| 345 | BF16 | VSA · FastH3 V2 · 80% | — | — | — | — |
| 345 | BF16 | Veda + LightX2V8 · 90% | — | — | — | — |
| 345 | BF16 | VDN · official | — | — | — | — |
| 345 | BF16 | Spark-H3 + LightX2V8 | — | — | — | — |
| 345 | INT8 | Full + LightX2V8 | — | — | — | — |
| 345 | INT8 | VSA · FastH3 V2 · 80% | — | — | — | — |
| 345 | INT8 | Veda + LightX2V8 · 90% | — | — | — | — |
| 345 | INT8 | VDN · official | — | — | — | — |
| 345 | INT8 | Spark-H3 + LightX2V8 | — | — | — | — |

Matched **Offload**: `omni-layer-prefetch1-vae-resident-v1`. Ratios use same-device/same-precision Full8. The higher speedup in each completed hardware pair is bold; ties and missing values are not. — means pending; old four-step or old-seed data cannot enter this table.

Validated: Ascend 0/20, PRO 6000 0/20.

[Raw eight-step records](benchmarks/single-card-highres.json)

<!-- END HIGHRES RESULTS -->


## Qualitative results and validation

Ascend 910B3 only: 1344×768, 345 native frames, 24 FPS (14.375 seconds, approximately 14.4 seconds), with audio. All five methods use eight forwards; BF16 is on the left and native W8A8 INT8 on the right. Samples use the reviewed common seed **7**; the remaining videos are being regenerated; earlier four-step or seed42 videos are not relabeled.

> A cyclist rides along a quiet green park path, smooth cinematic tracking shot, natural daylight.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Full + LightX2V8 · 8 steps**

New-seed eight-step video pending validation.

</td>
<td>

**Full + LightX2V8 · 8 steps**

New-seed eight-step video pending validation.

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 steps**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 steps**

New-seed eight-step video pending validation.

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 steps**

New-seed eight-step video pending validation.

</td>
<td>

**Veda + LightX2V8 · 90% · 8 steps**

New-seed eight-step video pending validation.

</td>
</tr>
<tr>
<td>

**VDN · official student · 8 steps**

New-seed eight-step video pending validation.

</td>
<td>

**VDN · official student · 8 steps**

New-seed eight-step video pending validation.

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 steps**

New-seed eight-step video pending validation.

</td>
<td>

**Spark-H3 + LightX2V8 · 8 steps**

New-seed eight-step video pending validation.

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

Common seed **7** was selected after reviewing 16 uniformly spaced frames, including the last frame, from two candidates: seed7 has one cyclist; seed123 retains two. Shot changes remain. Screening is user-requested and did not select the fastest timing; this screening request is excluded from formal measurements.

[New sample metadata and progress](examples/videos/qualitative-1344-345.json); [earlier seed/configuration archive](examples/videos/archive/qualitative-pre-unified8-seed42.json). Seed screening is user-requested and excluded from timing. Different students are not asserted to have pixel or perceptual equivalence.

The initial integration targets batch-one T2VA without reference videos.
See [architecture](docs/architecture.md), [validation status](docs/validation.md)
and [third-party notices](THIRD_PARTY_NOTICES.md).
