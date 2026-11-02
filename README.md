<p align="center">
  <img src="docs/assets/ascend-h3-logo.png" alt="Ascend-H3 logo" width="190">
</p>

<h1 align="center">Ascend-H3</h1>

<div align="center">

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/python-3.12-blue)](docs/environment.md)
[![OS: Linux](https://img.shields.io/badge/OS-Linux-lightgrey)](docs/environment.md)
[![Platform: Ascend 910B3](https://img.shields.io/badge/platform-Ascend%20910B3-0077C8)](docs/validation.md)
[![Backend: vLLM-Omni](https://img.shields.io/badge/backend-vLLM--Omni-orange)](docs/environment.md)
[![Email: liusonghua@sjtu.edu.cn](https://img.shields.io/badge/%E2%9C%89-Email-0077C8)](mailto:liusonghua@sjtu.edu.cn)

</div>

<p align="center"><a href="#install">Installation</a> | <a href="#models">Models</a> | <a href="#serve">Serve</a> | <a href="#performance">Performance</a></p>

<div align="center">

**English** | [简体中文](README.zh-CN.md) | [繁體中文](i18n/README.zh-TW.md) | [한국어](i18n/README.ko.md) | [Español](i18n/README.es.md) | [日本語](i18n/README.ja.md) | [हिन्दी](i18n/README.hi.md) | [मराठी](i18n/README.mr.md) | [Русский](i18n/README.ru.md) | [Português](i18n/README.pt.md) | [తెలుగు](i18n/README.te.md) | [Français](i18n/README.fr.md) | [Deutsch](i18n/README.de.md) | [Italiano](i18n/README.it.md) | [Tiếng Việt](i18n/README.vi.md) | [العربية](i18n/README.ar.md) | [اردو](i18n/README.ur.md) | [বাংলা](i18n/README.bn.md) | [فارسی](i18n/README.fa.md) | [Română](i18n/README.ro.md) | [Türkçe](i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](docs/assets/ascend-h3-ecosystem.png)

> **License:** Project-authored contributions in this distribution are offered under the [PolyForm Noncommercial License 1.0.0](LICENSE). Third-party code and model weights have separate terms. See [third-party notices](THIRD_PARTY_NOTICES.md). For uses not covered by the applicable terms, contact [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn).

MiniMax H3 inference on Ascend with Full, VSA, Veda, VDN and Spark-H3
attention in one repository. The integration reuses vLLM-Omni model loading,
INT8 transformer quantization and selectable tensor / sequence parallel inference.

**Validated scope:** 40/40 INT8 E2E cells at 1344×768, 124/345 frames and 1/2/4/8 NPUs (37 fresh measurements and 3 explicitly marked historical layer-Offload cells). The resident BF16 comparison on Ascend 910B3 and PRO 6000 contains 20 attention-core and 20 complete-block measurements. The attention operators retain their validated versions. An additional **30/30 fresh SP cells** cover both frame counts on 2/4/8 NPUs; all selected SP cells keep the DiT resident, with text-encoder Offload marked where used.

## Watch the Fair Comparison

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

[Parameters, SHA256 and provenance for all ten videos](examples/videos/qualitative-1344-345.json).

The initial integration targets batch-one T2VA without reference videos.
See [architecture](docs/architecture.md), [validation status](docs/validation.md)
and [third-party notices](THIRD_PARTY_NOTICES.md).

## Default 768p workloads

<table>
  <thead>
    <tr>
      <th width="200">Case</th>
      <th width="200">Width × height</th>
      <th width="200">Native output frames</th>
      <th width="200">Latent video grid</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>1344-124</code></td><td>1344×768</td><td>124</td><td>37×24×42</td></tr>
    <tr><td><code>1344-345</code></td><td>1344×768</td><td>345</td><td>102×24×42</td></tr>
  </tbody>
</table>

Output FPS is 24. Requests use explicit frame counts, with frame interpolation
disabled. The long case is the agreed native 345-frame workload; it is not
cropped to 341 frames. BF16 and INT8 use these two 1344×768 workloads. Omitting `--case` selects `1344-124`.

<a id="install"></a>

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

<a id="models"></a>

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

<a id="serve"></a>

## Start a server

Use idle NPUs and run one server per selected device set. Replace the example
paths with your downloaded [model assets](docs/weights.md).

**Single NPU** · Full attention with INT8 and text-encoder offload:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

**Choose a method** · Keep the command above and change `--method`, `--model`,
and any required arguments:

<table>
  <thead>
    <tr>
      <th width="120">Method</th>
      <th width="240"><code>--model</code></th>
      <th width="440">Additional arguments</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>full</code></td><td><code>/models/MiniMax-H3</code></td><td>—</td></tr>
    <tr><td><code>vsa</code></td><td><code>/models/FastH3-V2</code></td><td>None: official V2 full student weights, including trained gates, load through <code>--model</code>; do not pass <code>--lora</code>.</td></tr>
    <tr><td><code>veda</code></td><td><code>/models/MiniMax-H3</code></td><td><code>--predictor /models/veda/predictor.safetensors</code></td></tr>
    <tr><td><code>vdn</code></td><td><code>/models/MiniMax-H3</code></td><td><code>--lora /models/VDN/stage-dmd-step-250</code></td></tr>
    <tr><td><code>spark</code></td><td><code>/models/MiniMax-H3</code></td><td>—</td></tr>
  </tbody>
</table>

**Multiple NPUs** · Select the devices and choose tensor or sequence parallelism:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp
```

TP accepts 1, 2, 4, 7 or 8 devices and remains the default.
SP accepts 1, 2, 4 or 8 devices for all five methods; use `--parallel sp`
(`ulysses` is an alias). For example, four-card Spark supporting both frame
counts uses:

```bash
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel sp --steps 8 --offload text
```

See [SP launch commands and per-case memory settings](docs/serving.md#sequence-parallel-inference).
The default `--offload none` keeps weights resident. Use `--offload text` to
reduce text-encoder memory, or `--offload layer` to stream both the text encoder
and DiT. For single-NPU 345-frame Spark, VSA or VDN, use
`--offload layer --no-precompute-adaln`. Add `--dry-run` to inspect the native
command without loading a model.

For BF16, advanced offload options, AdaLN preparation, VAE parallelism and
request timeouts, see the [serving guide](docs/serving.md).

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

### BF16 latency at a glance

The charts visualize both tables above: attention core on the left and complete
DiT block on the right, for 124 and 345 frames. Bar labels show mean latency and
speedup against the Full baseline on the same device; these are layer measurements,
not end-to-end generation times.

![BF16 attention core and DiT block comparison](docs/figures/bf16-three-layer.png)

[All nine-run records and input identity checks](benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<a id="performance"></a>
<a id="sp-performance"></a>

<!-- BEGIN PARALLEL SCALING -->

## INT8 parallel latency and scaling: TP and SP

1344×768, eight actual DiT updates, seed 7. **Speedup = single-card E2E time ÷ multi-card E2E time**, using the same attention method, precision and frame count. TP and SP share the same measured single-card baseline. The single-card column shows baseline latency; each multi-card cell shows **E2E latency in seconds and speedup over that single-card baseline**.

<table>
  <thead>
    <tr>
      <th rowspan="2">Frames</th>
      <th rowspan="2">Method</th>
      <th rowspan="2" width="120" align="right">1-card latency</th>
      <th colspan="2" align="center">2 cards</th>
      <th colspan="2" align="center">4 cards</th>
      <th colspan="2" align="center">8 cards</th>
    </tr>
    <tr>
      <th width="120" align="right">TP</th>
      <th width="120" align="right">SP</th>
      <th width="120" align="right">TP</th>
      <th width="120" align="right">SP</th>
      <th width="120" align="right">TP</th>
      <th width="120" align="right">SP</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">124</td><td>Full</td><td align="right">197.66 s †T</td><td align="right">116.03 s<br>(1.70×)</td><td align="right">106.17 s †T<br>(1.86×)</td><td align="right">64.78 s<br>(3.05×)</td><td align="right">52.20 s<br>(3.79×)</td><td align="right">35.87 s<br>(5.51×)</td><td align="right">28.38 s<br>(6.96×)</td></tr>
    <tr><td align="right">124</td><td>VSA V2 · 80%</td><td align="right">146.81 s †T</td><td align="right">93.57 s<br>(1.57×)</td><td align="right">83.74 s †T<br>(1.75×)</td><td align="right">53.47 s<br>(2.75×)</td><td align="right">41.05 s<br>(3.58×)</td><td align="right">30.40 s<br>(4.83×)</td><td align="right">22.19 s<br>(6.61×)</td></tr>
    <tr><td align="right">124</td><td>Veda · 90%</td><td align="right">97.40 s †T</td><td align="right">69.66 s<br>(1.40×)</td><td align="right">59.09 s †T<br>(1.65×)</td><td align="right">43.38 s<br>(2.25×)</td><td align="right">29.59 s<br>(3.29×)</td><td align="right">25.33 s<br>(3.84×)</td><td align="right">17.22 s<br>(5.66×)</td></tr>
    <tr><td align="right">124</td><td>VDN · official</td><td align="right">216.91 s †T</td><td align="right">136.02 s<br>(1.59×)</td><td align="right">115.92 s †T<br>(1.87×)</td><td align="right">79.26 s<br>(2.74×)</td><td align="right">59.90 s<br>(3.62×)</td><td align="right">44.36 s<br>(4.89×)</td><td align="right">34.91 s<br>(6.21×)</td></tr>
    <tr><td align="right">124</td><td>Spark-H3 · 90%</td><td align="right">117.11 s †T</td><td align="right">79.41 s<br>(1.47×)</td><td align="right">67.14 s †T<br>(1.74×)</td><td align="right">48.87 s<br>(2.40×)</td><td align="right">35.02 s<br>(3.34×)</td><td align="right">29.05 s<br>(4.03×)</td><td align="right">20.78 s<br>(5.64×)</td></tr>
    <tr><td align="right">345</td><td>Full</td><td align="right">1211.17 s †T</td><td align="right">649.72 s †T<br>(1.86×)</td><td align="right">625.72 s †T<br>(1.94×)</td><td align="right">328.63 s<br>(3.69×)</td><td align="right">299.26 s<br>(4.05×)</td><td align="right">178.81 s<br>(6.77×)</td><td align="right">159.88 s<br>(7.58×)</td></tr>
    <tr><td align="right">345</td><td>VSA V2 · 80%</td><td align="right">617.55 s †L ‡R</td><td align="right">363.99 s †T<br>(1.70×)</td><td align="right">334.36 s †T<br>(1.85×)</td><td align="right">193.33 s<br>(3.19×)</td><td align="right">165.62 s<br>(3.73×)</td><td align="right">104.71 s<br>(5.90×)</td><td align="right">83.26 s<br>(7.42×)</td></tr>
    <tr><td align="right">345</td><td>Veda · 90%</td><td align="right">337.95 s †T</td><td align="right">225.31 s †T<br>(1.50×)</td><td align="right">188.40 s †T<br>(1.79×)</td><td align="right">130.11 s<br>(2.60×)</td><td align="right">98.70 s<br>(3.42×)</td><td align="right">75.93 s<br>(4.45×)</td><td align="right">54.98 s<br>(6.15×)</td></tr>
    <tr><td align="right">345</td><td>VDN · official</td><td align="right">620.67 s †L ‡R</td><td align="right">388.58 s †T<br>(1.60×)</td><td align="right">377.01 s †T<br>(1.65×)</td><td align="right">209.84 s<br>(2.96×)</td><td align="right">167.51 s †T<br>(3.71×)</td><td align="right">117.27 s<br>(5.29×)</td><td align="right">92.20 s<br>(6.73×)</td></tr>
    <tr><td align="right">345</td><td>Spark-H3 · 90%</td><td align="right">419.44 s †L ‡R</td><td align="right">267.61 s †T<br>(1.57×)</td><td align="right">231.16 s †T<br>(1.81×)</td><td align="right">150.50 s<br>(2.79×)</td><td align="right">120.20 s †T<br>(3.49×)</td><td align="right">87.02 s<br>(4.82×)</td><td align="right">66.18 s<br>(6.34×)</td></tr>
  </tbody>
</table>

†T: text-encoder Offload only; †L: DiT/text layer Offload; ‡R: reused validated historical measurement. Unmarked cells have no runtime Offload. All single-card baselines use text-only Offload except 345-frame VSA, VDN and Spark, which use DiT/text layer Offload and reuse fully validated measurements from 2026-10-31. Scaling therefore includes any change in Offload policy, rather than isolating parallel efficiency. All 30 SP runs keep the DiT resident.

Timing includes text encoding, video/audio decoding and MP4 writing, and excludes loading, prior LoRA merge, startup AdaLN preparation and warmups. The 30 SP cells each use one audited warmup and one formal request, with no extra profiling request. The TP/single-card source contains 37 fresh and 3 explicitly marked historical measurements. Selected-card contention invalidates an attempt and triggers a retry. VAE ranks match card count; DiT heads split 56/28/14/7.

[Single-card baselines and TP timing records](benchmarks/ascend-int8-e2e.json) · [SP timing records and source identity](benchmarks/ascend-sp-e2e.json).

SP2/SP4 reuse native W8A8 input quantization before head-local QKV projection where enabled; SP8 retains BF16 QKV transport. **VDN / SP2 / 345 frames** uses BF16 QKV exchange with `H3_ASCEND_INPUT_EXCHANGE=0` to avoid the extra head-weight cache; model linears remain INT8. Attention cores are unchanged. See the [SP launch commands and memory settings](docs/serving.md#sequence-parallel-inference).

<!-- END PARALLEL SCALING -->


The earlier matched layer-Offload comparison is retained in the [historical single-card reference](docs/archive/matched-offload-single-card.md). The timing JSON retains links to unchanged original records.
