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
| `vsa` | H3 base + Sol-H3 VSA student LoRA with trained compression gates | Logical KV64, selected KV blocks paired into KV128 on chip |
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
VSA accepts only its complete official FastH3 VSA student, including gates and
student weight updates; VDN accepts only its complete official stage-DMD
checkpoint. Neither accepts an additional community LoRA or student-disable flag.

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
h3-ascend serve --method vsa --model /models/MiniMax-H3 \
  --lora /models/Sol-H3-VSA-LoRA --precision bf16 --devices 1
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
Full/Veda/Spark/VSA, eight for VDN. For the optional original many-step model, use
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
includes fresh prompt encoding, all four denoising updates, video/audio
decoding, MP4 encoding and delivery. Full/Veda/Spark use the pinned LightX2V
four-step student; VSA uses its complete official FastH3 four-step student.
VDN requires its official eight-step configuration and is documented separately,
not mixed into the four-step figure. These students are not claimed to produce
identical quality, and the table makes no 50-to-four-step distillation claim.

Memory-constrained cases are measured with **Offload**, not listed as OOM.
For each cross-device pair, PRO 6000 and Ascend 910B3 must use the same explicit
resident/prefetch policy, including encoder and decoder residency. An Offload
marker and policy identifier accompany every affected row. BF16 and INT8
measurements identify their actual quantized modules and fallback layers.
The 1344×768, 345-frame figure compares both devices and both precisions.
Attention/FFN breakdowns come from a separate profiling run; they never replace
measured end-to-end time. Historical operator timings remain in raw benchmark
files for provenance and are not headline video speedups.

Speedups are reported only at **1344×768, 124/345 frames**. The 672×384
workloads remain supported for inference and functional validation.

Measured 345-frame single-card Ascend 910B3 rows, four steps, matched Offload,
seed 42, **one formal complete request per row**:

| Precision | Method | Complete video + audio | Speedup vs same-precision Full4 | BF16 / INT8 gain |
| --- | --- | ---: | ---: | ---: |
| BF16 | Full + LightX2V | 648.97 s | 1.00× | — |
| BF16 | Veda + LightX2V | 225.29 s | 2.88× | — |
| BF16 | Spark-H3 + LightX2V | 276.45 s | 2.35× | — |
| INT8 | Full + LightX2V | 603.84 s | 1.00× | 1.07× |
| INT8 | VSA · FastH3 | 261.56 s | 2.31× | — |
| INT8 | Veda + LightX2V | 203.77 s | 2.96× | 1.11× |

VSA BF16, Spark-H3 INT8, the 124-frame matrix and PRO 6000 results are
still pending. [Raw high-resolution records and independent profiles](benchmarks/single-card-four-step-highres.json).

[Measurement and figure protocol](docs/benchmark-protocol.md).

## Quality and validation

Full LightX2V four-step, native INT8, single Ascend 910B3, 672×384 / 124
frames, seed 42. The LoRA merged before quantization and inference. All four
DiT updates were finite, with all 50 blocks and INT8 QKV weights.

https://github.com/user-attachments/assets/8f1a4fc0-35a7-4fbd-a861-95fe3ca2cfb9

[MP4 with audio](examples/videos/full4-int8-single-672124.mp4) ·
[source, schedule and checksum](examples/videos/full4-int8-single-672124.json).
This is a qualitative sample, not a claim of universal perceptual equivalence.

VSA INT8, native TP4, 672×384, 124 frames, four-step trained VSA student:
all four ranks completed four DiT forwards with 50 nonzero gates each step.
The complete pipeline produced the expected video and audio.

https://github.com/user-attachments/assets/590be000-8cbd-458e-9198-afb2a2bdcaa4

[MP4 with audio](examples/videos/vsa-int8-tp4-672124.mp4) ·
[sample contract](examples/videos/vsa-int8-tp4-672124.json).
Frame-by-frame inspection found a brief brightness dip around 3.4 seconds
in this VSA INT8 sample. The same-seed BF16 sample also dips at that position;
it is not exclusive to INT8. [Same-seed BF16 MP4](examples/videos/vsa-bf16-tp4-672124.mp4)
and [its contract](examples/videos/vsa-bf16-tp4-672124.json) are included.
[Temporal diagnostics](benchmarks/vsa-temporal-sample-check.json)
record the observations; decoder versus latent causality remains unresolved.
Generation checks passing do not establish flicker-free or equivalent quality.
This is a qualitative smoke test, not a Full-versus-VSA quality evaluation.
All four workloads have now passed complete VSA INT8 TP4 inference and
decode; [the audited matrix](benchmarks/vsa-int8-tp4-matrix.json) records
exact frame counts and video hashes.

Veda INT8 TP4, 672×384, 124 frames, Turbo8, seed 42, nominal 90% sparsity:

https://github.com/user-attachments/assets/93ff95b6-615c-4fc0-bf84-bb7dabe6b08a

[MP4 with audio](examples/videos/veda-int8-tp4-672124.mp4) ·
[sample contract](examples/videos/veda-int8-tp4-672124.json).
All four Veda INT8 TP4 workloads passed complete generation and decode;
see [the audited matrix](benchmarks/veda-int8-tp4-matrix.json).

Previous strict Spark experiments generated 1344×768 videos at 124 and 345
frames with every decoded uint8 pixel matching the previous Ascend reference.
That is an Ascend-to-Ascend implementation check, not equality with CUDA or
Full attention. It must be requalified after framework integration.
Previous VDN tests produced plausible short clips. Numerical inversion is
approximate; the automatic BF16 temporal-convolution path also has a different
feature-rounding boundary from the upstream FP32 tap loop. See
[architecture](docs/architecture.md) for the FP32 temporal option.
No universal perceptual-equivalence claim is made.

The initial integration targets batch-one T2VA without reference videos.
See [architecture](docs/architecture.md), [validation status](docs/validation.md)
and [third-party notices](THIRD_PARTY_NOTICES.md).
