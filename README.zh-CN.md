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

<p align="center"><a href="#install">安装</a> | <a href="#models">模型</a> | <a href="#serve">启动服务</a> | <a href="#performance">性能结果</a></p>

<div align="center">

[English](README.md) | **简体中文** | [繁體中文](i18n/README.zh-TW.md) | [한국어](i18n/README.ko.md) | [Español](i18n/README.es.md) | [日本語](i18n/README.ja.md) | [हिन्दी](i18n/README.hi.md) | [मराठी](i18n/README.mr.md) | [Русский](i18n/README.ru.md) | [Português](i18n/README.pt.md) | [తెలుగు](i18n/README.te.md) | [Français](i18n/README.fr.md) | [Deutsch](i18n/README.de.md) | [Italiano](i18n/README.it.md) | [Tiếng Việt](i18n/README.vi.md) | [العربية](i18n/README.ar.md) | [اردو](i18n/README.ur.md) | [বাংলা](i18n/README.bn.md) | [فارسی](i18n/README.fa.md) | [Română](i18n/README.ro.md) | [Türkçe](i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](docs/assets/ascend-h3-ecosystem.png)

> **许可说明：**本发行版本中的项目自有贡献采用 [PolyForm Noncommercial 1.0.0 非商用许可](LICENSE)。第三方代码和模型权重适用各自的条款，之前授予的 Apache-2.0 权利不受影响，详见[第三方许可说明](THIRD_PARTY_NOTICES.md)。超出适用条款的使用可联系 [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn)。

将 MiniMax H3 的 Full、VSA、Veda、VDN、Spark-H3 昇腾注意力实现
整合到同一个仓库，复用 vLLM-Omni 的模型加载、INT8 量化和 attention
head 多卡并行，并支持可选的张量并行（TP）与序列并行（SP）。

**已验证范围：**1344×768、124/345 帧、1/2/4/8 卡的 INT8 端到端共 40/40 项，其中 37 项本轮新测、3 项明确标记的历史分层 Offload 实测。昇腾 910B3 与 PRO 6000 的 BF16 常驻对照包含 20 项注意力核心和 20 项完整 DiT block 测量；注意力算子保留此前已验收版本。 新增 **30/30 组新测 SP 结果**，覆盖双／四／八卡的两种时长；全部 DiT 常驻，使用文本编码器 Offload 的格子单独标记。

## 默认 768p 视频规格

| case | 空间分辨率 | 原生输出帧数 | 视频 token 网格 |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

输出 24 FPS。请求直接指定帧数，关闭插帧。长视频保留原生 345 帧，
不裁剪成 341 帧。BF16 和 INT8 均使用这两组 1344×768 规格；省略 `--case` 时默认生成 124 帧。

<a id="install"></a>

## 安装与编译

使用 Linux、Python 3.12 及匹配的 Ascend Torch/CANN 环境。
vLLM、vLLM-Ascend、vLLM-Omni 的固定版本和验收状态见
[环境说明](docs/environment.md)，避免混装不同版本。

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

编译不需要占用 NPU，生成的动态库放在仓库外的缓存目录中。
只编译一种方法时，将 `all` 换成 `vsa`、`veda` 或 `spark`。
Full 和 VDN 不需要这里的自定义 CANN 扩展。当前使用源码可编辑安装，
安装后保留 checkout 中的 `configs/` 和 `csrc/`。

<a id="models"></a>

## 模型和算法

| 方法 | 所需权重 | 注意力机制 |
| --- | --- | --- |
| `full` | H3，默认 LightX2V 8-step，也可使用社区 LoRA | 全注意力 |
| `vsa` | 官方 FastH3 V2 完整学生权重，含 50 个训练后的 compression gate | 8 步，80% 视频稀疏率；逻辑 KV64，两个选中 KV64 在片上配对为 KV128 |
| `veda` | H3 + LightX2V 8-step + 官方训练的 Veda predictor | 每层、每个 head 动态选块，默认名义稀疏度 90% |
| `vdn` | H3 + 官方 VDN stage-DMD 权重目录 | local softmax + 保留 128 维状态的 DeltaNet 分支 |
| `spark` | H3 + LightX2V 8-step | 内容相关层次分块，原 token 与 proxy 联合 softmax，精确分支默认名义稀疏度 90% |

模型权重需按原项目许可自行下载，不包含在本仓库中。不同蒸馏模型
具有不同的权重和采样规则；不能认为任意权重直接换算子就能保证画质。
VSA 必须加载训练后的非零 gate；VDN 必须加载其学习得到的分支权重。
官方权重下载及准备命令见[模型准备](docs/weights.md)。

社区 LoRA 可先准备为加载时融合的格式（不指定 `--steps` 则不固定采样步数）：

```bash
h3-ascend prepare-lora --source /models/style-diffusers-scaled.safetensors \
  --output /models/style
```

已经折入 alpha/rank 的文件用默认 `diffusers-scaled`，原生 H3 命名用 `raw-h3`，
带 alpha 元数据的 PEFT 文件用 `peft`；禁止重复缩放。官方 LightX2V 默认文件的
准备由启动器自动完成，规范化 `.lora_A.default.weight` 等键并核验固定 SHA256。

Full、Veda、Spark 默认启用固定版本 **LightX2V 8-step LoRA**。
`--no-few-step-lora` 关闭默认少步学生；配合 Full 可运行原版多步模型（本轮报告基线为八步）。
可用 `--few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors`
提供本地官方文件，否则自动按固定版本下载并核验 SHA256。
社区风格 LoRA 先用 `prepare-lora` 准备，再通过 `--lora` 与默认学生在加载时合并。
指定另一个社区少步学生时关闭默认 LightX2V，避免叠加两个学生模型。
Veda 始终必须带官方训练好的预测网络。VSA 只接受官方 FastH3 V2 完整学生权重，
采用 8 次前向、80% 视频稀疏率、视频/音频 shift 10/3，不传 `--lora`；
VDN 只接受官方 stage-DMD 配置，二者不叠加其他 LoRA，也不支持关闭其学生模型。
全部 LoRA 在计时前、TP 切分及 INT8 量化前合并，推理时不额外执行 LoRA 矩阵乘。

<a id="serve"></a>

## 启动服务

只使用没有其他计算进程的卡。单卡 Full：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

其他注意力：

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

同一组卡一次只启动一个服务。加入 `--dry-run` 可查看生成的原生
vLLM-Omni 命令，不会加载权重。

单卡示例流式加载文本编码器，INT8 DiT 常驻 NPU。原生配方只量化 DiT，
文本编码器仍是 BF16，启动策略需要一并考虑它的显存占用。
**BF16：** 使用 `--precision bf16`；内存不足时显式选择
`--offload layer --no-precompute-adaln`。INT8 使用 vLLM-Omni 原生量化配方：

```json
{"transformer":{"method":"int8"}}
```

INT8 投影输出的 BF16 Q/K/V 进入注意力。注意力累加、Veda predictor、
VDN 状态计算保留各自规定的精度，不能将其描述成“注意力全程 INT8”。
原生配方会将不支持的层保留为 BF16。例如单卡 H3 的一项投影输出宽度
为 96,768，超过量化矩阵乘的 65,535 限制；TP 可以缩小每卡宽度，
让相应层进入可量化范围。

**四卡：** 默认采用 vLLM 原生 TP 的 attention head 切分：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# 八卡：每个 rank 划分七个 DiT attention head。
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices` 支持 1、2、4、7、8，均可整除 56 个 head。
Veda 在分片后仍使用原 checkpoint 的全局 head 编号。
七卡 DiT TP 时，文本编码器使用一张卡，因为其 64 个 query head、
8 个 KV head 不能按七份切分。
启动默认 `--offload none`，权重常驻设备。内存不足时，`--offload text` 只流式
加载文本编码器，DiT 仍常驻；`--offload layer` 同时流式加载两个组件，每次保留
一个计算块和一个异步预取块。`--offload module` 另提供整个组件切换，正在验收。
下方已完成的统一 Offload 对照明确使用
`--offload layer`（`omni-layer-prefetch1-vae-resident-v1`），两种硬件策略相同。
新一轮端到端**只测试 INT8**，优先常驻，必要的 Offload 回退会单独标记。
双方 attention/DiT block 对照统一 BF16，取第 0/24/49 层，每层跑三次，
九次耗时之和除以九得到每层平均时延。
五种注意力均可选择 `--parallel tp` 或 `--parallel sp`；`ulysses` 保留为
`sp` 的兼容别名，默认仍为 TP。SP 支持单／双／四／八卡；TP 另支持七卡。
原生 VAE 分块解码现在默认使用全部选中卡（`--vae-parallel-size 0`），
可用 `--vae-parallel-size 1` 改为单个解码 rank。INT8 Full、124 帧的
单卡、双卡、四卡、八卡 VAE 完整链路已通过；新增 SP 的五种方法、两种帧数、双／四／八卡共 30 组也已全部完成验收。

NPU 的 DiT 常驻模式会在服务就绪前准备 AdaLN 结果。准备过程读取实际加载的
模型及已合并 LoRA，解析生效的 sigma 序列、DiT 更新次数及视频／音频 shift，
并非固定八步。可用 `serve --steps N` 指定原模型或自定义学生的启动步数，
客户端对应使用 `generate --steps N`；与蒸馏模型自带计划冲突时会报错。
缓存容量随所需计划增长，投影沿用原生计算，AdaLN 主权重留在 CPU，准备好的
结果保存在 NPU。四／八卡另按精确 timestep 输入缓存原生时间嵌入，避免 FP32
归约的末位变化导致后续缓存失效；权重版本、数值环境和各 rank 命中一致性
仍会检查。更换采样配置后重启即可重新准备；未准备的输入会走原始计算，
可能重新搬运权重。`--no-precompute-adaln` 可关闭准备；DiT 整组件／逐层
Offload 默认关闭此功能。

Spark/VSA/VDN 的单卡 345 帧使用 `--offload layer --no-precompute-adaln`，
保留此前已验收的原算子。端到端表以 †L 标记原生分层 Offload，‡R 标记
从上一轮复用的同配置实测；其他格子的常驻策略按各自记录说明。

启动器将原生工作进程的单次请求等待上限设为 7200 秒，避免长视频被上游默认的
600 秒上限中止。可在启动前设置 `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` 覆盖；这只
调整等待上限，不改变模型计算或请求计时。

### 序列并行推理

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel sp --steps 8 --offload text

# VDN 保留官方学生模型与原始 128 维 linear 分支。
h3-ascend serve --method vdn --model /models/MiniMax-H3 \
  --lora /models/VDN/stage-dmd-step-250 \
  --precision int8 --devices 4 --parallel sp --steps 8 --offload text
```

SP 将 FFN 和残差运算按 token 行分配，保留每卡完整的 DiT 主权重。
INT8 双／四卡的默认常驻路径交换原生 W8A8 本来就需要的输入激活
及 FP32 缩放系数，再只计算当前 rank 负责的 heads。head 权重缓存从已合并
LoRA、已量化的权重生成。注意力核心及 BF16 输出回传沿用原实现，不额外引入
有损通信量化。该路径的每个 rank 在请求准备时缓存完整 RoPE 表，供所有层和步复用。
八卡输入 AllGather 实测更慢，因此仍交换 BF16 QKV；BF16 推理和 DiT 流式
加载也保留原路径。可用 `H3_ASCEND_INPUT_EXCHANGE=0` 在双／四卡选择原 SP 路径。

VSA 保留原门控投影，门控单独重分布。启用前置投影路径时，VDN 的 linear 分支复用已准备好的
完整序列、局部 heads 的原始 QKV，保留 BF16 分支输入、原始特征映射、求逆、
扫描和读出。VDN 的逐位核验以同一 SP head 划分为参照；head 划分本身与单次
完整 heads 的分支调用可能存在浮点差异。


这些 SP 启动示例采用 `text`，便于同时请求两种帧数。实测策略是：双卡全部
仅文本 Offload；四卡 124 帧全部 `none`，345 帧的 Full／VSA／Veda 用 `none`、
VDN／Spark 用 `text`；八卡全部 `none`。VAE 默认使用全部选中卡，DiT 不逐层
卸载。VDN 双卡 345 帧还须设置 `H3_ASCEND_INPUT_EXCHANGE=0`，以去掉额外
的 head 权重缓存；模型保留原有 INT8 配方。完整启动方式见
[SP 服务指南](docs/serving.md#sequence-parallel-inference)。

<a id="performance"></a>
<a id="sp-performance"></a>

<!-- BEGIN SP E2E MATRIX -->

## 新测 INT8 SP：双卡／四卡／八卡

1344×768、8 次实际 DiT 更新、seed 7。每个设定为**一次带对齐检查的预热、一次正式请求**，不追加 profile 请求，30 格均为新测。计时包括文本编码、视频／音频解码与 MP4 写入，排除模型加载、提前合并 LoRA 和启动阶段的 AdaLN 准备。每格列出 SP 耗时及同卡数原 TP 耗时除以 SP 耗时的比值；这是完整链路对照，不是纯通信加速比。

| 帧数 | 方法 | SP2 · 相对 TP2 | SP4 · 相对 TP4 | SP8 · 相对 TP8 |
| ---: | --- | ---: | ---: | ---: |
| 124 | Full | 106.17 s †T · 1.09× | 52.20 s · 1.24× | 28.38 s · 1.26× |
| 124 | VSA V2 · 80% | 83.74 s †T · 1.12× | 41.05 s · 1.30× | 22.19 s · 1.37× |
| 124 | Veda · 90% | 59.09 s †T · 1.18× | 29.59 s · 1.47× | 17.22 s · 1.47× |
| 124 | VDN · official | 115.92 s †T · 1.17× | 59.90 s · 1.32× | 34.91 s · 1.27× |
| 124 | Spark-H3 · 90% | 67.14 s †T · 1.18× | 35.02 s · 1.40× | 20.78 s · 1.40× |
| 345 | Full | 625.72 s †T · 1.04× | 299.26 s · 1.10× | 159.88 s · 1.12× |
| 345 | VSA V2 · 80% | 334.36 s †T · 1.09× | 165.62 s · 1.17× | 83.26 s · 1.26× |
| 345 | Veda · 90% | 188.40 s †T · 1.20× | 98.70 s · 1.32× | 54.98 s · 1.38× |
| 345 | VDN · official | 377.01 s †T · 1.03× | 167.51 s †T · 1.25× | 92.20 s · 1.27× |
| 345 | Spark-H3 · 90% | 231.16 s †T · 1.16× | 120.20 s †T · 1.25× | 66.18 s · 1.31× |

†T：仅文本编码器 Offload；†L：DiT／文本逐层 Offload；†M：整组件 Offload。无标记的 SP 格子不使用运行时 Offload。TP 沿用下表各自记录的策略，因此策略不同时，比值也包含 Offload 差异。选中卡出现竞争时，该次记录作废并重跑。VAE 并行卡数与总卡数一致。双／四卡在启用时复用原生输入量化；各原始记录注明实际投影通信路径。八卡及流式路径保留 BF16 QKV 通信。注意力核心不变。

[新30组记录与源码标识](benchmarks/ascend-sp-e2e.json)

DiT 常驻、采用 BF16 QKV 通信的例外：**VDN · official / SP2 / 345**。在对应启动命令前加 `H3_ASCEND_INPUT_EXCHANGE=0`，并使用表中记录的 Offload 选项；这样去掉额外的 head 投影权重缓存，模型线性层仍为 INT8。

<!-- END SP E2E MATRIX -->

## 生成两组 768p 视频

先启动服务，再创建 UTF-8 prompt 文件并发送请求：

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

客户端传入与服务端相同的 `--method`：Full/VSA/Veda/VDN/Spark 均默认八步。
如需运行原版多步模型，服务端与客户端均加 `--no-few-step-lora`，客户端默认 50 步。
独立社区学生模型可显式指定 `--steps`；`--guidance-scale` 省略时保留原生策略。
本项目不额外添加 Refine。客户端保存完整 MP4 和请求/时延 JSON。

<!-- BEGIN BF16 LAYER RESULTS -->

## BF16 三层对照：双方均常驻

仅 1344×768，124/345 帧。真实学生权重、逐字节共享的固定合成隐藏状态；不是实际视频请求的逐层激活。第 0、24、49 层各预热两次、正式运行三次，**九次耗时之和除以九**。双方均不 Offload；BF16 激活/主权重，保留算法要求的 FP32 预测网络、累积与 VDN 状态。加速比分母均为同硬件、同形状的 BF16 Full。每组相对加速比更高者加粗；绝对时延也同时列出。

### 仅注意力核心

从已准备好的 Q/K/V 开始，到注意力输出结束；排除 QKV 投影、Q/K norm、RoPE、输出投影、AdaLN 和 FFN。计入稀疏方法必需的 gate、预测、选块、重排与还原；VDN 计入 softmax 与 linear 两条完整核心分支。所有方法的主表均使用这一口径，不以完整 attention 模块耗时替代。核心计时已完成 10/10 组跨硬件对照，待测项不沿用旧模块数字。

| 帧数 | 方法 | 昇腾时延 | Pro 6000 时延 | 昇腾加速比 | Pro 6000 加速比 |
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

### 完整 DiT block

从 block 输入到 block 输出，包含投影、归一化、RoPE、注意力核心、AdaLN、FFN 和残差。与核心独立测量，不能把差值当作 FFN 时延，也不能外推端到端时间。

| 帧数 | 方法 | 昇腾时延 | Pro 6000 时延 | 昇腾加速比 | Pro 6000 加速比 |
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

[原始九次数据及输入校验信息](benchmarks/bf16-three-layer.json)。

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

## TP 参考：INT8 单卡、双卡、四卡、八卡

1344×768，八次实际 DiT 更新，端到端包含文本编码、音视频解码与 MP4 写出；不含加载、提前 LoRA merge、AdaLN 准备和预热。优先让 DiT 常驻；常驻组正式请求前后逐 rank 核验 AdaLN 零未命中与零权重搬运。明确标记的单卡 345 帧组在实际显存不足后改用原生分层 Offload，并按用户要求复用配置一致、已完整验收的历史实测；逐项标记来源，不视为新测。

| 帧数 | 方法 | 单卡 | 双卡 | 四卡 | 八卡 |
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

已覆盖 **40/40**：本轮新测 37 组，复用历史实测 3 组。†T：仅文本编码器逐层 Offload；†L：DiT 与文本编码器原生逐层 Offload，预取一层，VAE 常驻；‡R：复用 2026-10-31 的同配置原始记录。未标记格子无运行期 Offload。仅常驻组预计算 AdaLN；逐层 Offload 组不声称 DiT 常驻或 AdaLN 零搬运。VAE 并行数等于卡数，DiT head 按 56/28/14/7 分配；保留原生 INT8 覆盖规则。

[INT8 E2E records](benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

此前统一分层 Offload 的对照移至[历史单卡附录](docs/archive/matched-offload-single-card.zh-CN.md)。当前表中复用的格子仍链接到未改动的原始记录。

## 效果展示与验证范围

仅展示昇腾 910B3 的 **1344×768、345 原生帧、24 FPS（14.375 秒，约 14.4 秒）**完整音视频。

五种方法全部八步；左列 BF16，右列原生 W8A8 INT8。所有样例单卡、相同 prompt、共同种子 **7**、同一 Offload 策略。

> A cyclist rides along a quiet green park path, smooth cinematic tracking shot, natural daylight.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Full + LightX2V8 · 8 步**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Full + LightX2V8 · 8 步**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 步**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 步**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 步**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 步**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · official · 8 步**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · official · 8 步**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 步**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 步**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 为 80% 视频稀疏率；Veda 为 90%。文本和音频保留各自原生规则。

[十个视频的参数、SHA256 与来源](examples/videos/qualitative-1344-345.json)。已完整解码核对 345 帧及音轨，并检查每段 16 个均匀抽样帧。不同学生不保证像素或感知等价；VSA 仍有镜头切换，不作无闪烁保证。

首版目标是 batch=1、无参考视频的 T2VA。更多信息见
[架构设计](docs/architecture.md)、[验证状态](docs/validation.md)
及[第三方许可](THIRD_PARTY_NOTICES.md)。
