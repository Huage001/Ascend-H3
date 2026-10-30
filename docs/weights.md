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

## Default four-step Full / Veda / Spark

The launcher defaults to the pinned official
[LightX2V Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
`minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`, revision
`30f485ed874aa44c8183348ad361c5eabad318da`, SHA256
`1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c`.
The contract is four steps, video shift 6, audio shift 3, alpha=rank=128.
Download is automatic, or provide `serve --few-step-source FILE`.
The source's PEFT default-namespace keys are normalized and alpha/rank is
folded into the adapter once. Weights merge before TP and INT8 quantization.

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --revision 30f485ed874aa44c8183348ad361c5eabad318da --local-dir /models/lightx2v
# Original model: disable the default student at both endpoints.
h3-ascend serve --method full --model /models/MiniMax-H3 --no-few-step-lora
h3-ascend generate --method full --no-few-step-lora --case 1344-345 \
  --prompt-file prompt.txt --output original-50step.mp4
```

General community adapters use `prepare-lora --source FILE --output DIR`.
Use `diffusers-scaled` for already-scaled matrices, `raw-h3` for official
Diffusers H3 conversion, or `peft` for default-namespace matrices with declared
alpha metadata. Partial Q/K/V adapters are supported. Omit `--steps` for style
adapters; a separately distilled student can pin `--steps`, `--video-shift`,
`--audio-shift`. Do not combine two different distilled schedules.
A legacy eight-step Turbo adapter remains an explicit optional input; it is
not substituted for the default LightX2V four-step source.

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
