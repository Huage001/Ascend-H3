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

## Default eight-step Full / Veda / Spark

The launcher defaults to the pinned official
[LightX2V Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
`minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors`, revision
`30f485ed874aa44c8183348ad361c5eabad318da`, SHA256
`9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4`.
The contract is eight actual forwards, video shift 6, audio shift 3, alpha=rank=128.
Download is automatic, or provide `serve --few-step-source FILE`.
The source's PEFT default-namespace keys are normalized and alpha/rank is
folded into the adapter once. Weights merge before TP and INT8 quantization.

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors \
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
Older four-step LightX2V and historical eight-step adapters are not defaults.
Current Full/Veda/Spark all use the exact pinned 8-step 768p source in
`configs/lightx2v8.json`; VSA and VDN retain their own official eight-step students.

## FastH3 V2 VSA: eight forwards, 80% sparsity

Use the complete official [FastH3 V2 student](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2),
revision `3da2ddfe1954d9cda4c05b643dc0f26007a655c5`. This is a **full checkpoint**,
not the old v1 LoRA. Do not stack LightX2V or another adapter. The released
contract is eight actual DiT forwards (nine sigma points), tile64, 80% nominal
video sparsity, guidance 1, video/audio shifts **10/3**. Base sigma positions are
`[0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125, 0]`; Omni applies the
trained shifts. Its API uses `--steps 8`, not the raw FastVideo grid count 9.

The downloader pins every file digest in `configs/fasth3-v2.json`. It can reuse
existing text/processor/tokenizer and native VAE files **only after hash checks**;
it always obtains the entire trained V2 transformer and all 50 gates.
The transformer download alone is 70.10 GB. If frozen components are absent,
additional storage is required. The native VAE compatibility revision is pinned
to the one required by the selected Omni release.

```bash
python scripts/download_fasth3_v2.py --output /models/FastH3-V2 \
  --base /models/MiniMax-H3 --native-cache /models/hf-v2 \
  --endpoint https://huggingface.co --workers 4
# Use --endpoint https://hf-mirror.com when required by your network.
export HF_HUB_CACHE=/models/hf-v2 HF_HUB_OFFLINE=1
h3-ascend serve --method vsa --model /models/FastH3-V2 \
  --precision bf16 --devices 1
h3-ascend generate --method vsa --case 1344-345 --prompt-file prompt.txt \
  --output vsa-v2.mp4
```

中文：VSA 已切换到官方 FastH3 V2 完整学生模型，80% 稀疏、8 次前向、shift 10/3。
不能仅把旧 LoRA 的稀疏度从 90% 改为 80%，也不能复用旧四步结果；完整权重、训练门控、
采样日程必须一起更换。INT8 使用同一 V2 学生，经原生量化后运行；双卡/四卡仍使用
vLLM-Omni attention-head TP。新版视频和性能未验收前均标记为待完成。

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
