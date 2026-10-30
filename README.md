# Ascend-H3

[中文说明](README.zh-CN.md)

MiniMax H3 inference on Ascend with Full, VSA, Veda, VDN and Spark-H3
attention in one repository. The integration reuses vLLM-Omni model loading,
INT8 transformer quantization and native tensor parallel attention heads.

**Integration validation is in progress.** The source operators have compiled
successfully; this branch is not yet a qualified release of every
method × precision × workload × parallel configuration. Historical speed
measurements below are explicitly separate from validation of this repository.

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
| `full` | Native MiniMax H3 model; optionally a matching Turbo LoRA | Dense attention |
| `vsa` | H3 base + Sol-H3 VSA student LoRA with trained compression gates | Logical KV64, selected KV blocks paired into KV128 on chip |
| `veda` | H3 base + matching Turbo LoRA + Veda predictor | Dynamic per-head Veda block routes, nominal 90% sparsity |
| `vdn` | H3 base + official VDN stage-DMD checkpoint directory | Local softmax plus 128-dimensional learned DeltaNet branch |
| `spark` | H3 base + matching Turbo LoRA | Content-dependent Spark-H3 hierarchical partition, nominal 90% sparsity |

Download models under their original licenses. Weights are not redistributed.
Different distilled students have different schedules and learned rules;
swapping attention on arbitrary weights does not guarantee matching quality.
VSA requires nonzero trained gates. VDN requires its learned branch weights.
Download and preparation commands are in [model assets](docs/weights.md).

For the previously exported **already-scaled, 363-target Diffusers Turbo8**
adapter (including AdaLN), create an explicit manifest first:

```bash
h3-ascend prepare-lora --source /models/turbo8-diffusers.safetensors \
  --output /models/H3-Turbo-LoRA --steps 8 --video-shift 12 --audio-shift 3
```

This records the artifact hash and schedule and links its weight file. It
does not apply alpha scaling a second time. Marked adapters fuse during normal
checkpoint loading, before head sharding and INT8 quantization, with ordinary
layerwise offload. Keep the linked weight file available. Raw LightX2V
artifacts use the native loader and must not be marked as already-scaled.

## Start a server

Use only devices that have no other compute processes. For one card:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision bf16 --devices 1
```

Other backends use the same server API:

```bash
h3-ascend serve --method vsa --model /models/MiniMax-H3 \
  --lora /models/Sol-H3-VSA-LoRA --precision bf16 --devices 1
h3-ascend serve --method veda --model /models/MiniMax-H3 \
  --lora /models/H3-Turbo-LoRA --predictor /models/veda/predictor.safetensors \
  --precision bf16 --devices 1
h3-ascend serve --method vdn --model /models/MiniMax-H3 \
  --lora /models/VDN/stage-dmd-step-250 --precision bf16 --devices 1
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --lora /models/H3-Turbo-LoRA --precision bf16 --devices 1
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
  --lora /models/H3-Turbo-LoRA --precision int8 --devices 4 --parallel tp
```

`--devices` supports 1, 2, 4, 7 and 8, which divide 56 heads. Veda retains
global predictor head IDs after sharding. Seven-card DiT TP uses one text
encoder rank because its 64 query / 8 KV heads cannot be divided by seven.
VSA, VDN and explicitly marked legacy
adapters use ordinary layerwise offload for load-time weight fusion. Other
paths use distributed layerwise offload with 16 resident DiT layers. The text
encoder is also selected for layerwise offload. Optional `--parallel ulysses`
is exposed for non-VDN paths; native TP is the default requested strategy.

## Generate the four cases

Create a UTF-8 prompt file, start a server, then use its matching step count:

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 672-124 672-345 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --steps 8 --seed 42 --output "outputs/${CASE}.mp4"
done
```

Use `--steps 4` for the matching four-step Sol-H3 student. VDN stage-DMD
requires eight steps. Use the base model's native schedule for non-distilled
Full inference; `--guidance-scale` is optional and otherwise leaves the
model's native guidance policy in control. No extra refinement stage is added by this package.
The client writes the video plus a JSON request/timing receipt.

## Performance evidence

Speedup means **Full time / candidate time**, with both measured on the same
device. Values below are historical single-card measurements of representative
layers 0/24/49, BF16 activations, nominal 90% sparse exact branches. They are
not end-to-end video speedups or measurements of the new unified integration.

Attention dispatch includes routing, permutation, sparse computation and output
restoration, with QKV already available; it excludes QKV/output projections:

| Workload | Ascend VSA | Ascend Veda | Ascend Spark | PRO6000 VSA | PRO6000 Veda | PRO6000 Spark |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 672×384, 124 | 1.14× | 1.34× | — | 0.97× | 0.84× | 0.88× |
| 672×384, 345 | 1.59× | 2.52× | — | 1.55× | 2.47× | 1.89× |
| 1344×768, 124 | 2.22× | 4.10× | 2.49× | 2.23× | 4.45× | 2.65× |
| 1344×768, 345 | 3.40× | 6.64× | 4.12× | 3.51× | 6.73× | 4.36× |

VDN has learned projections and a second branch. Its separate measurements
include the **complete attention module**, including QKV/output projections:

| Workload | Ascend VDN128 / Full | PRO6000 official VDN128 / Full |
| --- | ---: | ---: |
| 672×384, 124 | 0.49× | 0.61× |
| 672×384, 345 | 0.85× | 1.04× |
| 1344×768, 124 | 1.01× | 1.14× |
| 1344×768, 345 | 2.35× | 2.63× |

Do not rank VDN against the dispatch-only table. A speedup below 1 means
slower than Full. Ascend and NVIDIA use their own optimized Full baselines;
equal ratios do not imply equal absolute latency. See
[machine-readable historical evidence](benchmarks/historical-results.json).
Missing entries remain unmeasured, including unified INT8/multi-card speedups.

The PRO6000 VSA column was measured on 2026-10-30 with official Triton
tile64, layer 24, ten interleaved repetitions and a precomputed nonzero
compression gate. Its raw records are in
[the VSA reference](benchmarks/pro6000-vsa64-20261030.json). The older Ascend
VSA/Veda experiments used synthetic QKV; NVIDIA Veda/Spark and Ascend Spark
used real-weight projected synthetic activations. These cross-device ratios
are reference points rather than identical-QKV comparisons.

## Quality and validation

Full INT8, native TP2, 672×384, 124 frames, Turbo8, seed 42 — generated through
this integration with video and audio. Both ranks completed eight DiT forwards,
with all 50 blocks per step, finite outputs and INT8 QKV weights.

![Full INT8 TP2 example](examples/videos/full-int8-tp2-672124.gif)

[MP4 with audio](examples/videos/full-int8-tp2-672124.mp4) ·
[sample contract and checksum](examples/videos/full-int8-tp2-672124.json).
This is one qualitative sample; audit synchronization makes it unsuitable as
a performance measurement. It does not establish perceptual equivalence.

VSA INT8, native TP4, 672×384, 124 frames, four-step trained VSA student:
all four ranks completed four DiT forwards with 50 nonzero gates each step.
The complete pipeline produced the expected video and audio.

![VSA INT8 TP4 example](examples/videos/vsa-int8-tp4-672124.gif)

[MP4 with audio](examples/videos/vsa-int8-tp4-672124.mp4) ·
[sample contract](examples/videos/vsa-int8-tp4-672124.json).
Frame-by-frame inspection found a brief brightness dip around 3.4 seconds
in this VSA INT8 sample. Its cause is pending the BF16 comparison.
Generation checks passing do not establish flicker-free or equivalent quality.
This is a qualitative smoke test, not a Full-versus-VSA quality evaluation.
All four workloads have now passed complete VSA INT8 TP4 inference and
decode; [the audited matrix](benchmarks/vsa-int8-tp4-matrix.json) records
exact frame counts and video hashes.

Veda INT8 TP4, 672×384, 124 frames, Turbo8, seed 42, nominal 90% sparsity:

![Veda INT8 TP4 example](examples/videos/veda-int8-tp4-672124.gif)

[MP4 with audio](examples/videos/veda-int8-tp4-672124.mp4) ·
[sample contract](examples/videos/veda-int8-tp4-672124.json).
All four Veda INT8 TP4 workloads passed complete generation and decode;
see [the audited matrix](benchmarks/veda-int8-tp4-matrix.json).

Previous strict Spark experiments generated 1344×768 videos at 124 and 345
frames with every decoded uint8 pixel matching the previous Ascend reference.
That is an Ascend-to-Ascend implementation check, not equality with CUDA or
Full attention. It must be requalified after framework integration.
Previous VDN tests produced plausible short clips; numerical inversion is
approximate. No universal perceptual-equivalence claim is made.

The initial integration targets batch-one T2VA without reference videos.
See [architecture](docs/architecture.md), [validation status](docs/validation.md)
and [third-party notices](THIRD_PARTY_NOTICES.md).
