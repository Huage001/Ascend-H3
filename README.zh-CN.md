# Ascend-H3

[English](README.md)

将 MiniMax H3 的 Full、VSA、Veda、VDN、Spark-H3 昇腾注意力实现
整合到同一个仓库，复用 vLLM-Omni 的模型加载、INT8 量化和 attention
head 多卡并行。

**当前仍在进行整合验收。** 自定义 CANN 算子已从源码编译通过；
尚未将所有“方法 × 精度 × 视频规格 × 并行规模”标记为通过。
完整 DiT block 已完成 20/20 项，注意力核心已完成 20/20 项；新 INT8 常驻优先端到端矩阵已验收 4/40。此前统一 Offload 的 40 组单卡八步记录仅作历史参考。

## 默认 768p 视频规格

| case | 空间分辨率 | 原生输出帧数 | 视频 token 网格 |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

输出 24 FPS。请求直接指定帧数，关闭插帧。长视频保留原生 345 帧，
不裁剪成 341 帧。BF16 和 INT8 均使用这两组 1344×768 规格；省略 `--case` 时默认生成 124 帧。

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

## 启动服务

只使用没有其他计算进程的卡。单卡 Full：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision bf16 --devices 1
```

其他注意力：

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

同一组卡一次只启动一个服务。加入 `--dry-run` 可查看生成的原生
vLLM-Omni 命令，不会加载权重。

**INT8：** 任意方法将 `--precision bf16` 改为 `--precision int8`。
底层使用 vLLM-Omni 原生量化配方：

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
  --precision bf16 --devices 8 --parallel tp
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
非 VDN 方法另提供 `--parallel ulysses`；默认使用 TP。
`--vae-parallel-size 0` 另可将原生 VAE 分块解码分配到全部选中卡，默认值为 1。
原生 VAE 四卡、124 帧完整链路已通过；八卡仍在验收。

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

## INT8 端到端：昇腾单卡、双卡、四卡、八卡

1344×768，八次实际 DiT 更新，含文本编码、视频/音频解码及 MP4 写出；模型加载、预先 LoRA merge 与预热不计时。按实际显存依次尝试常驻、仅文本编码器 Offload、整组件切换、逐层 Offload；只有记录了真实 OOM 才回退。下表只填完整请求、全部 rank 审计与独立 profile 通过的结果。

| 帧数 | 方法 | 单卡 | 双卡 | 四卡 | 八卡 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 124 | Full | 196.31 s †L | 117.38 s †T | 64.94 s | — |
| 124 | VSA V2 · 80% | — | — | — | — |
| 124 | Veda · 90% | — | — | 43.39 s | — |
| 124 | VDN · official | — | — | — | — |
| 124 | Spark-H3 · 90% | — | — | — | — |
| 345 | Full | — | — | — | — |
| 345 | VSA V2 · 80% | — | — | — | — |
| 345 | Veda · 90% | — | — | — | — |
| 345 | VDN · official | — | — | — | — |
| 345 | Spark-H3 · 90% | — | — | — | — |

已验收 **4/40**。— 为待测。†T：仅文本编码器逐层 Offload；†M：整组件切换；†L：DiT 与文本编码器逐层 Offload。未标记的格子不 Offload。VAE 使用与设备数相同的原生 tile 并行；DiT head 按 56/28/14/7 分配。原生 INT8 仅量化受支持线性层，单卡过宽 AdaLN 的 BF16 回退及多卡覆盖变化保留在原始配置中。

[INT8 E2E records](benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

## 历史：统一 Offload 的单卡端到端对照

**所有方法、基线和计时请求统一为 8 次实际 DiT 前向。** Full/Veda/Spark 使用固定版本的官方 LightX2V 8-step 768p LoRA（shift 6/3）；VSA 使用官方 FastH3 V2（80% 稀疏率、shift 10/3）；VDN 保留官方八步学生。Veda 保持 90% 稀疏率。各学生保留自己的训练日程，不把四步权重强行运行八步。

加速比 = 同一硬件、同一分辨率/帧数/种子的 **Full BF16 八步端到端耗时 / 当前方法八步端到端耗时**。BF16 与 INT8 均使用这个 BF16 Full 分母，Full INT8 也一样。报告仅覆盖 1344×768 的 124 帧和 345 帧，单卡，BF16 / INT8；先合并 LoRA，再进行 TP 切分与量化。计时包含文本编码、全部八次去噪、音视频解码、MP4 编码和完整返回；模型加载、权重合并、预热及画面筛选不计时。

VSA 已按用户要求选定共同种子 **7**，各方法正式测速使用同一个 prompt 和种子；筛选过程已记录。下表只接受这套新口径的实测值，不把旧四步结果翻倍或换算。[此前 Full 四步为基线的数据](benchmarks/archive/pre-unified8-full4-reference.json)仅作历史归档。

<!-- BEGIN HIGHRES RESULTS -->

全部八步；共同种子：7

| 帧数 | 精度 | 方法 | 昇腾 910B3 耗时 | 昇腾加速比 | PRO 6000 耗时 | PRO 6000 加速比 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 124 | BF16 | Full + LightX2V8 | 206.71 秒 | 1.00× | 91.20 秒 | 1.00× |
| 124 | BF16 | VSA · FastH3 V2 · 80% | 163.20 秒 | **1.27×** | 79.20 秒 | 1.15× |
| 124 | BF16 | Veda + LightX2V8 · 90% | 113.43 秒 | **1.82×** | 53.60 秒 | 1.70× |
| 124 | BF16 | VDN · official | 301.42 秒 | 0.69× | 102.39 秒 | **0.89×** |
| 124 | BF16 | Spark-H3 + LightX2V8 | 132.78 秒 | **1.56×** | 61.81 秒 | 1.48× |
| 124 | INT8 | Full + LightX2V8 | 189.57 秒 | **1.09×** | 85.06 秒 | 1.07× |
| 124 | INT8 | VSA · FastH3 V2 · 80% | 146.89 秒 | **1.41×** | 74.92 秒 | 1.22× |
| 124 | INT8 | Veda + LightX2V8 · 90% | 97.61 秒 | **2.12×** | 48.28 秒 | 1.89× |
| 124 | INT8 | VDN · official | 213.95 秒 | **0.97×** | 97.74 秒 | 0.93× |
| 124 | INT8 | Spark-H3 + LightX2V8 | 116.19 秒 | **1.78×** | 54.38 秒 | 1.68× |
| 345 | BF16 | Full + LightX2V8 | 1179.84 秒 | 1.00× | 467.81 秒 | 1.00× |
| 345 | BF16 | VSA · FastH3 V2 · 80% | 661.43 秒 | **1.78×** | 318.03 秒 | 1.47× |
| 345 | BF16 | Veda + LightX2V8 · 90% | 376.99 秒 | **3.13×** | 173.62 秒 | 2.69× |
| 345 | BF16 | VDN · official | 842.14 秒 | 1.40× | 299.42 秒 | **1.56×** |
| 345 | BF16 | Spark-H3 + LightX2V8 | 472.93 秒 | **2.49×** | 196.73 秒 | 2.38× |
| 345 | INT8 | Full + LightX2V8 | 1125.43 秒 | **1.05×** | 454.32 秒 | 1.03× |
| 345 | INT8 | VSA · FastH3 V2 · 80% | 617.55 秒 | **1.91×** | 304.01 秒 | 1.54× |
| 345 | INT8 | Veda + LightX2V8 · 90% | 338.55 秒 | **3.48×** | 157.90 秒 | 2.96× |
| 345 | INT8 | VDN · official | 620.67 秒 | **1.90×** | 281.81 秒 | 1.66× |
| 345 | INT8 | Spark-H3 + LightX2V8 | 419.44 秒 | **2.81×** | 182.20 秒 | 2.57× |

统一 **Offload**：`omni-layer-prefetch1-vae-resident-v1`。BF16 与 INT8 的加速比都相对各自硬件的 Full BF16 八步；低于 1× 表示慢于该基线。每组对照中更高的加速比加粗，缺失或相同的值不加粗。— 为待测；旧四步或旧种子结果不进入新表。

已验收：昇腾 20/20, PRO 6000 20/20.

[八步原始记录](benchmarks/single-card-highres.json)

<!-- END HIGHRES RESULTS -->

这里是双方均为八步的完整请求加速比，不包含从 50 步降到 8 步的蒸馏收益，
也不是单独注意力内核的加速比。注意力稀疏化不会减少 FFN、编码器、解码器和
MP4 编码的工作，统一的 Offload 策略也始终启用。INT8 量化覆盖受支持的
Transformer 线性层，不代表所有注意力运算都使用 INT8。昇腾的相对加速比
更高不代表绝对耗时低于 PRO 6000，请同时比较耗时列。VSA V2 使用 80%
稀疏率，Veda 和 Spark 使用 90%；这些行比较的是不同注意力机制和学生模型，
不能理解为相同质量或相同稀疏率下的纯内核对照。

此前端到端模块分解图保留为[历史诊断数据](benchmarks/long-video-e2e.json)；其中较宽的模块计时不进入上面的注意力核心对照。

[此前统一逐层 Offload 的 BF16 八卡诊断](docs/archive/tp8-matched-offload.md)，不并入新 INT8 常驻优先矩阵。


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
