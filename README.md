# Ascend-H3

[中文说明](README.zh-CN.md)

MiniMax H3 inference on Ascend with Full, VSA, Veda, VDN and Spark-H3
attention in one repository. The integration reuses vLLM-Omni model loading,
INT8 transformer quantization and selectable tensor / sequence parallel inference.

**Validated scope:** 40/40 INT8 E2E cells at 1344×768, 124/345 frames and 1/2/4/8 NPUs (37 fresh measurements and 3 explicitly marked historical layer-Offload cells). The resident BF16 comparison on Ascend 910B3 and PRO 6000 contains 20 attention-core and 20 complete-block measurements. The attention operators retain their validated versions.

## Default 768p workloads

| Case | Width × height | Native output frames | Latent video grid |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

Output FPS is 24. Requests use explicit frame counts, with frame interpolation
disabled. The long case is the agreed native 345-frame workload; it is not
cropped to 341 frames. BF16 and INT8 use these two 1344×768 workloads. Omitting `--case` selects `1344-124`.

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
  --precision int8 --devices 1 --offload text
```

Other backends use the same server API:

```bash
h3-ascend serve --method vsa --model /models/FastH3-V2 \
  --precision int8 --devices 1 --offload text
h3-ascend serve --method veda --model /models/MiniMax-H3 \
  --predictor /models/veda/predictor.safetensors \
  --precision int8 --devices 1 --offload text
h3-ascend serve --method vdn --model /models/MiniMax-H3 \
  --lora /models/VDN/stage-dmd-step-250 --precision int8 --devices 1 --offload text
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

Run one server at a time on the selected devices. Add `--dry-run` to inspect
the generated native vLLM-Omni command without loading a model.

The single-card examples stream the text encoder while keeping the INT8 DiT
resident. The native recipe quantizes the DiT; the text encoder remains BF16,
so its memory footprint still needs to be included in the launch policy.
For **BF16**, use `--precision bf16`; if necessary select
`--offload layer --no-precompute-adaln` explicitly. INT8 uses the native recipe:

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

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices` supports 1, 2, 4, 7 and 8, which divide 56 heads. Veda retains
global predictor head IDs after sharding. Seven-card DiT TP uses one text
encoder rank because its 64 query / 8 KV heads cannot be divided by seven.
Launches default to `--offload none` (resident weights). If memory is insufficient,
`--offload text` streams only the text encoder while retaining the DiT on device;
`--offload module` swaps complete components (experimental qualification);
`--offload layer` streams both components, keeping one active and one prefetched
block. The completed matched-Offload reference below explicitly uses
`--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) on both hardware
platforms. New end-to-end measurements use **INT8 only**, starting with resident
execution and marking every necessary Offload fallback. The isolated attention
and DiT-block comparison uses BF16 on both devices: layers 0/24/49, three runs
each, summing the nine timings and dividing by nine. Optional
Select `--parallel tp` or `--parallel sp` for all five attention methods.
`ulysses` remains an alias for `sp`; TP remains the default. SP supports
1/2/4/8 cards, while TP additionally supports seven cards.
Native tiled VAE decoding now defaults to all selected cards
(`--vae-parallel-size 0`); use `--vae-parallel-size 1` for a single decoder rank.
The INT8 Full / 124-frame pipeline has passed with one, two, four and eight
VAE ranks. Other method/frame-count combinations remain in the E2E matrix.

Resident NPU launches prepare AdaLN results before server readiness. Preparation
resolves the loaded model and merged LoRA's actual sigma positions, DiT update
count, and video/audio shifts; it is not hard-coded to eight steps. Use
`serve --steps N` to specify a teacher/custom-student launch, and use the same
count in `generate --steps N`. A conflicting distilled checkpoint schedule is
rejected. Cache capacity grows with the required plans. The original projection
math is retained; AdaLN master weights stay on CPU, while prepared results stay
on NPU. TP4/TP8 also memoize native time embeddings from exact timestep inputs
to avoid low-bit FP32 reduction variation invalidating downstream caches.
Weights, numerical settings and all-rank hit agreement still guard reuse.
Restart with the new configuration to precompute a different schedule; an
unprepared input uses the original computation and may transfer weights.
`--no-precompute-adaln` disables preparation; DiT module/layer streaming keeps it
disabled by default.

For single-card 345-frame Spark/VSA/VDN, use `--offload layer --no-precompute-adaln`
with the original validated operators. The E2E table marks these native layer
Offload cells with †L; ‡R identifies matching measurements reused from the
previous run. All other cells retain their individually recorded residency policy.

The launcher sets the native worker request deadline to 7200 seconds so a long
video is not aborted by the upstream 600-second default. Set
`VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` before launch to override this limit; it changes
waiting behavior, not model computation or the measured request duration.

### Sequence parallel inference

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel sp --steps 8 --offload none

# VDN keeps its official student and original 128-dimensional linear branch.
h3-ascend serve --method vdn --model /models/MiniMax-H3 \
  --lora /models/VDN/stage-dmd-step-250 \
  --precision int8 --devices 4 --parallel sp --steps 8 --offload none
```

SP distributes token rows through FFN and residual operations while retaining
replicated DiT master weights. For resident INT8 SP2/SP4, the QKV path exchanges
the input activation and FP32 scale already required by native W8A8, then projects
only this rank's heads. Head-local weight caches are prepared after LoRA fusion
and quantization. Attention cores and BF16 output redistribution are unchanged;
there is no additional lossy communication quantization. On this path every rank
prepares a complete RoPE table once per request and shares it across layers and steps.
SP8 retains BF16 QKV redistribution because input AllGather was slower at that
degree. BF16 inference and DiT streaming also retain the original QKV path.
Set `H3_ASCEND_INPUT_EXCHANGE=0` to request that reference path on SP2/SP4.

VSA preserves the original gate projection and redistributes its gate separately.
VDN reuses the full-sequence head-local raw QKV for its original linear branch,
retaining BF16 branch inputs, original features/inverse/scans, and readout.
VDN comparisons qualify against the same SP head partition; head partitioning
itself can have floating-point differences from a single full-head branch call.

<!-- BEGIN SP E2E MATRIX -->

<!-- Fresh30 SP results are inserted here only after all guarded cases pass. -->

<!-- END SP E2E MATRIX -->

## Generate the two 768p cases

Create a UTF-8 prompt file, start a server, then use its matching step count:

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

Pass the server's `--method` to the client: all five methods
(Full/VSA/Veda/VDN/Spark) default to eight actual forwards. For the optional original many-step model, use
`generate --method full --no-few-step-lora` (50 steps), matching the server.
An explicit `--steps` remains available for a separately prepared community
student. `--guidance-scale` is optional; omission preserves native policy.
No Refine stage is added. The client saves the MP4 and a JSON timing receipt.

<!-- BEGIN BF16 LAYER RESULTS -->

## BF16 layer comparison: resident on both devices

1344×768, 124/345 frames only. Real student weights and byte-identical frozen synthetic hidden states; these are not layer activations captured from a video request. Layers 0, 24 and 49 each receive two untimed warmups and three formal runs: **sum the nine timings and divide by nine**. Weights stay resident on both devices. Activations/main weights are BF16; required FP32 predictors, accumulations and VDN state are retained. Speedups use the same-hardware, same-shape BF16 Full baseline. The higher relative speedup is bold; absolute latency is shown too.

### Attention core only

Starts with prepared Q/K/V and ends at the attention output. Excludes QKV projection, Q/K norm, RoPE, output projection, AdaLN and FFN. Includes required sparse gates, prediction, selection, rearrangement and restoration; VDN includes both complete softmax and linear cores. Every method uses this boundary; whole-module timings are not substituted. Core timing has completed 10/10 matched hardware pairs; pending cells do not reuse earlier module measurements.

| Frames | Method | Ascend latency | PRO 6000 latency | Ascend speedup | PRO 6000 speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 124 | Full | 343.31 ms | 111.64 ms | 1.00× | 1.00× |
| 124 | VSA V2 · 80% | 216.24 ms | 93.56 ms | **1.59×** | 1.19× |
| 124 | Veda · 90% | 75.50 ms | 25.01 ms | **4.55×** | 4.46× |
| 124 | VDN · official | 368.89 ms | 142.74 ms | **0.93×** | 0.78× |
| 124 | Spark-H3 · 90% | 128.54 ms | 39.35 ms | 2.67× | **2.84×** |
| 345 | Full | 2578.60 ms | 838.10 ms | 1.00× | 1.00× |
| 345 | VSA V2 · 80% | 1196.30 ms | 498.34 ms | **2.16×** | 1.68× |
| 345 | Veda · 90% | 380.26 ms | 123.45 ms | 6.78× | **6.79×** |
| 345 | VDN · official | 1093.46 ms | 423.12 ms | **2.36×** | 1.98× |
| 345 | Spark-H3 · 90% | 636.98 ms | 184.38 ms | 4.05× | **4.55×** |

### Complete DiT block

Runs from block input to block output, including projections, normalization, RoPE, attention core, AdaLN, FFN and residual operations. Measured independently of the core; their difference is not an FFN measurement, and neither is extrapolated into E2E time.

| Frames | Method | Ascend latency | PRO 6000 latency | Ascend speedup | PRO 6000 speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 124 | Full | 491.51 ms | 185.93 ms | 1.00× | 1.00× |
| 124 | VSA V2 · 80% | 346.40 ms | 166.56 ms | **1.42×** | 1.12× |
| 124 | Veda · 90% | 213.56 ms | 97.76 ms | **2.30×** | 1.90× |
| 124 | VDN · official | 527.75 ms | 222.89 ms | **0.93×** | 0.83× |
| 124 | Spark-H3 · 90% | 267.42 ms | 112.65 ms | **1.84×** | 1.65× |
| 345 | Full | 3000.60 ms | 1054.58 ms | 1.00× | 1.00× |
| 345 | VSA V2 · 80% | 1550.07 ms | 703.92 ms | **1.94×** | 1.50× |
| 345 | Veda · 90% | 787.91 ms | 330.60 ms | **3.81×** | 3.19× |
| 345 | VDN · official | 1534.19 ms | 647.19 ms | **1.96×** | 1.63× |
| 345 | Spark-H3 · 90% | 1028.43 ms | 393.06 ms | **2.92×** | 2.68× |

![BF16 attention core and DiT block comparison](docs/figures/bf16-three-layer.png)

[All nine-run records and input identity checks](benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

## INT8 E2E: Ascend 1 / 2 / 4 / 8 cards

1344×768, eight actual DiT updates, including text encoding, video/audio decoding and MP4 writing. Loading, prior LoRA merge, AdaLN preparation and warmups are excluded. Prefer resident DiT; resident rows verify zero AdaLN misses and weight transfers on every rank around the formal request. After recorded OOMs, the marked single-card 345-frame cells use native layer Offload and reuse matching, fully validated prior measurements as requested; these cells are explicitly marked rather than presented as fresh measurements.

| Frames | Method | 1 card | 2 cards | 4 cards | 8 cards |
| ---: | --- | ---: | ---: | ---: | ---: |
| 124 | Full | 197.66 s †T | 116.03 s | 64.78 s | 35.87 s |
| 124 | VSA V2 · 80% | 146.81 s †T | 93.57 s | 53.47 s | 30.40 s |
| 124 | Veda · 90% | 97.40 s †T | 69.66 s | 43.38 s | 25.33 s |
| 124 | VDN · official | 216.91 s †T | 136.02 s | 79.26 s | 44.36 s |
| 124 | Spark-H3 · 90% | 117.11 s †T | 79.41 s | 48.87 s | 29.05 s |
| 345 | Full | 1211.17 s †T | 649.72 s †T | 328.63 s | 178.81 s |
| 345 | VSA V2 · 80% | 617.55 s †L ‡R | 363.99 s †T | 193.33 s | 104.71 s |
| 345 | Veda · 90% | 337.95 s †T | 225.31 s †T | 130.11 s | 75.93 s |
| 345 | VDN · official | 620.67 s †L ‡R | 388.58 s †T | 209.84 s | 117.27 s |
| 345 | Spark-H3 · 90% | 419.44 s †L ‡R | 267.61 s †T | 150.50 s | 87.02 s |

Covered **40/40**: 37 fresh and 3 reused prior measurements. †T: text-encoder layer Offload only; †L: native DiT/text-encoder layer Offload with one-layer prefetch and resident VAE; ‡R: matching original measurements from 2026-10-31. Unmarked cells have no runtime Offload. Only resident rows precompute AdaLN; layer-Offload rows make no DiT-residency or zero-AdaLN-transfer claim. VAE ranks match card count; DiT heads split 56/28/14/7. Native INT8 coverage is retained.

[INT8 E2E records](benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

The earlier matched layer-Offload comparison is retained in the [historical single-card reference](docs/archive/matched-offload-single-card.md). Reused cells in the current table link to unchanged original records.

## Qualitative results and validation

Ascend 910B3 only: **1344×768, 345 native frames, 24 FPS (14.375 seconds, approximately 14.4 seconds)**, with audio.

All five methods use eight forwards. Left: BF16. Right: native W8A8 INT8. All samples use one card, the same prompt, common seed **7** and matched Offload.

> A cyclist rides along a quiet green park path, smooth cinematic tracking shot, natural daylight.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Full + LightX2V8 · 8 steps**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Full + LightX2V8 · 8 steps**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 steps**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 steps**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 steps**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 steps**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · official · 8 steps**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · official · 8 steps**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 steps**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 steps**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 uses 80% video sparsity; Veda uses 90%. Text and audio retain their native rules.

[Parameters, SHA256 and provenance for all ten videos](examples/videos/qualitative-1344-345.json). Full decoding verifies 345 frames and audio; 16 uniform frames per clip were visually reviewed. Distinct students are not asserted to be pixel- or perceptually equivalent. VSA retains shot changes; no flicker-free guarantee is made.

The initial integration targets batch-one T2VA without reference videos.
See [architecture](docs/architecture.md), [validation status](docs/validation.md)
and [third-party notices](THIRD_PARTY_NOTICES.md).
