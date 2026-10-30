# Model assets / 模型准备

Keep weights outside Git and use their original licenses. The commands below
use the official Hugging Face CLI. Use a mirror endpoint if required by your
network. 模型不随源码分发；下载到容量足够的模型盘。

## H3 base

[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) supplies the FL2VA
backbone, text encoder, tokenizer, processor and both VAEs. T2VA runs through
this partition. Keep its native layout rather than mixing a Diffusers DiT
with native component metadata.

```bash
hf download MiniMaxAI/MiniMax-H3 --include 'model_index.json' 'FL2VA/*' \
  --local-dir /models/MiniMax-H3
```

## Eight-step Full / Veda / Spark teacher

Download the original [Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora),
then use the official Diffusers H3 converter and an explicit eight-step manifest:

```bash
hf download larryvrh/MiniMax-H3-Turbo-Lora minimax_h3_turbo_v4_step600_ema.safetensors \
  --local-dir /models/turbo-source
h3-ascend prepare-lora \
  --source /models/turbo-source/minimax_h3_turbo_v4_step600_ema.safetensors \
  --source-format raw-h3 --output /models/H3-Turbo-LoRA \
  --steps 8 --video-shift 12 --audio-shift 3
```

`raw-h3` converts and writes a self-contained scaled adapter. The default
`diffusers-scaled` links an existing converted file and must not be used for
raw input. Both fuse before TP slicing and online INT8 quantization.
原始权重使用 `raw-h3`；已经转换并折入缩放的旧文件使用默认格式，避免重复缩放。
With Diffusers 0.40.0, all 726 converted tensors matched the earlier handoff
bit for bit; the complete converted file SHA256 also matches.

## Four-step VSA

Use a trained VSA student rather than adding an untrained compression gate:
[FastH3 VSA adapter family](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/tree/main/vsa-datafree).

```bash
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  vsa-datafree/adapter_model.safetensors --local-dir /models/fasth3
# serve --lora /models/fasth3/vsa-datafree/adapter_model.safetensors
# generate --steps 4
```

## Veda predictor

[Published Veda bundle](https://github.com/veda-sparse/Miowtion#weights)
includes the predictor and per-head tile plans:

```bash
hf download Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview \
  --local-dir /models/veda
ln -s minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors \
  /models/veda/predictor.safetensors
```

The FP8 storage is decoded by the bundle loader; prediction retains the
original FP32 scoring arithmetic. The 672×384 case rebuilds valid token tiles
for that geometry while reusing the published head-shape choices; it is outside
the plan's 1344×768 search geometry. 小分辨率沿用 head 的 tile 形状选择，
重新构建有效 token 布局；不宣称已重新搜索或微调。

## VDN student

[Official VDN checkpoint](https://huggingface.co/OpenVDN/vdn-minimax-h3):

```bash
hf download OpenVDN/vdn-minimax-h3 --include 'stage-dmd-step-250/*' \
  --local-dir /models/VDN
# serve --lora /models/VDN/stage-dmd-step-250
# generate --steps 8
```

Keep the full stage directory, including its specification, adapters and
learned linear branch. 不能只加载 Turbo LoRA 而省略 VDN 的学习分支。
