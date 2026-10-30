# Ascend-H3

[English](README.md)

将 MiniMax H3 的 Full、VSA、Veda、VDN、Spark-H3 昇腾注意力实现
整合到同一个仓库，复用 vLLM-Omni 的模型加载、INT8 量化和 attention
head 多卡并行。

**当前仍在进行整合验收。** 自定义 CANN 算子已从源码编译通过；
尚未将所有“方法 × 精度 × 视频规格 × 并行规模”标记为通过。
下文历史性能数据与这个仓库的整合验收分开记录。

## 四组视频规格

| case | 空间分辨率 | 原生输出帧数 | 视频 token 网格 |
| --- | --- | --- | --- |
| `672-124` | 672×384 | 124 | 37×12×21 |
| `672-345` | 672×384 | 345 | 102×12×21 |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

输出 24 FPS。请求直接指定帧数，关闭插帧。长视频保留原生 345 帧，
不裁剪成 341 帧。BF16 和 INT8 均使用这四组规格。

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
| `full` | H3，默认 LightX2V 4-step，也可使用社区 LoRA | 全注意力 |
| `vsa` | H3 + 带训练后 compression gate 的 Sol-H3 VSA LoRA | 逻辑 tile64，选中的两个 KV64 在片上配对成 KV128 计算 |
| `veda` | H3 + LightX2V 4-step + 官方训练的 Veda predictor | 每层、每个 head 动态选块，默认名义稀疏度 90% |
| `vdn` | H3 + 官方 VDN stage-DMD 权重目录 | local softmax + 保留 128 维状态的 DeltaNet 分支 |
| `spark` | H3 + LightX2V 4-step | 内容相关层次分块，原 token 与 proxy 联合 softmax，精确分支默认名义稀疏度 90% |

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

Full、Veda、Spark 默认启用固定版本 **LightX2V 4-step LoRA**。
`--no-few-step-lora` 关闭默认少步学生；配合 Full 可运行原版多步模型（本轮报告基线为四步）。
可用 `--few-step-source /models/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`
提供本地官方文件，否则自动按固定版本下载并核验 SHA256。
社区风格 LoRA 先用 `prepare-lora` 准备，再通过 `--lora` 与默认学生在加载时合并。
指定另一个社区少步学生时关闭默认 LightX2V，避免叠加两个学生模型。
Veda 始终必须带官方训练好的预测网络。VSA 只接受完整 FastH3 VSA 官方 LoRA，
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
```

`--devices` 支持 1、2、4、7、8，均可整除 56 个 head。
Veda 在分片后仍使用原 checkpoint 的全局 head 编号。
七卡 DiT TP 时，文本编码器使用一张卡，因为其 64 个 query head、
8 个 KV head 不能按七份切分。
所有方法统一使用 **Offload：一个计算块加一个异步预取块**。DiT 和文本编码器的
权重保存在锁页 CPU 内存中，VAE 常驻设备。策略编号为
`omni-layer-prefetch1-vae-resident-v1`，N 卡对照必须使用同一固定版本的 Omni
实现和相同策略，避免为不同方法设置不同的常驻层数。
非 VDN 方法另提供 `--parallel ulysses`；默认使用 TP。

## 生成四组视频

先启动服务，再创建 UTF-8 prompt 文件并发送请求：

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 672-124 672-345 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 42 --output "outputs/${CASE}.mp4"
done
```

客户端传入与服务端相同的 `--method`：Full/Veda/Spark/VSA 默认四步，VDN 八步。
如需运行原版多步模型，服务端与客户端均加 `--no-few-step-lora`，客户端默认 50 步。
独立社区学生模型可显式指定 `--steps`；`--guidance-scale` 省略时保留原生策略。
本项目不额外添加 Refine。客户端保存完整 MP4 和请求/时延 JSON。

## 端到端性能

新的单卡矩阵正在实测，仅下方“已测”数据已完成，其余仍待测。最新口径为相同硬件、精度、分辨率、
帧数下，**Full + LightX2V 四步**端到端耗时 / 对应四步注意力方案端到端耗时。
量化收益另报同一方案的 BF16 / INT8 耗时比。此前 50 步基线已按要求停止，
不用于本轮图表。

模型加载、LoRA 合并、JIT 预热均在计时前完成；计时包含每次文本编码、四次去噪、
音视频解码、MP4 编码及返回。Full、Veda、Spark 使用固定版本 LightX2V 4-step；
VSA 使用完整官方 FastH3 4-step 学生。固定八步的 VDN 单独说明，不混入四步候选图。
不同学生不宣称画质等价，也不将 50→4 步蒸馏的收益计入本轮加速比。

显存不足的设定使用 **Offload** 实测，不以 OOM 作为最终结果。PRO 6000 与昇腾
910B3 的每组对照必须使用一致的驻留、预取及编码器/解码器加载策略，并在表格和
图中注明 Offload 和策略编号。BF16、INT8 均记录实际量化模块与 BF16 回退层。
1344×768、345 帧的图将展示两种硬件量化前后的各注意力结果。
Attention/FFN 分项来自单独的性能分析运行，不能代替端到端实测。历史算子数据
保留在原始记录中，不作为这里的视频生成加速比。

加速比只报告 **1344×768、124/345 帧**两组；672×384 保留推理和功能验证支持。

<!-- BEGIN HIGHRES RESULTS -->

Ascend 910B3 · 1344×768 · 单卡 · 四步 · seed 42，**每行一次正式完整请求**：

| 帧数 | 精度 | 方法 | 完整视频 + 音频 | 相对同精度 Full4 | BF16 / INT8 | 内存策略 |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 124 | BF16 | Full + LightX2V | 121.04 秒 | 1.00× | — | Offload |
| 124 | BF16 | VSA · FastH3 | 83.72 秒 | 1.45× | — | Offload |
| 124 | BF16 | Veda + LightX2V | 69.86 秒 | 1.73× | — | Offload |
| 124 | BF16 | Spark-H3 + LightX2V | 78.15 秒 | 1.55× | — | Offload |
| 124 | INT8 | Full + LightX2V | 111.90 秒 | 1.00× | 1.08× | Offload |
| 124 | INT8 | VSA · FastH3 | 74.24 秒 | 1.51× | 1.13× | Offload |
| 124 | INT8 | Veda + LightX2V | 61.78 秒 | 1.81× | 1.13× | Offload |
| 124 | INT8 | Spark-H3 + LightX2V | 71.41 秒 | 1.57× | 1.09× | Offload |
| 345 | BF16 | Full + LightX2V | 648.97 秒 | 1.00× | — | Offload |
| 345 | BF16 | VSA · FastH3 | 285.69 秒 | 2.27× | — | Offload |
| 345 | BF16 | Veda + LightX2V | 225.29 秒 | 2.88× | — | Offload |
| 345 | BF16 | Spark-H3 + LightX2V | 276.45 秒 | 2.35× | — | Offload |
| 345 | INT8 | Full + LightX2V | 603.84 秒 | 1.00× | 1.07× | Offload |
| 345 | INT8 | VSA · FastH3 | 261.56 秒 | 2.31× | 1.09× | Offload |
| 345 | INT8 | Veda + LightX2V | 203.77 秒 | 2.96× | 1.11× | Offload |
| 345 | INT8 | Spark-H3 + LightX2V | 247.17 秒 | 2.44× | 1.12× | Offload |

RTX PRO 6000 · 1344×768 · 单卡 · 四步 · seed 42，**每行一次正式完整请求**：

| 帧数 | 精度 | 方法 | 完整视频 + 音频 | 相对同精度 Full4 | BF16 / INT8 | 内存策略 |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 345 | BF16 | Full + LightX2V | 250.88 秒 | 1.00× | — | Offload |

已完整验收：昇腾 16/16 组，PRO 6000 1/16 组。 其余组合仍在补齐，不使用预热或失败记录作为测量结果。

[高分辨率原始记录和独立分项测量](benchmarks/single-card-four-step-highres.json)。

<!-- END HIGHRES RESULTS -->













详见[端到端计时、Offload 和图表口径](docs/benchmark-protocol.md)。

## 效果与验证范围

默认 LightX2V 四步，Full、原生 INT8、单卡昇腾 910B3，672×384 / 124 帧、seed 42。
LoRA 在量化和推理前合并；四次 DiT 前向输出有限，每步经过全部 50 层，QKV 权重为 INT8。

https://github.com/user-attachments/assets/8f1a4fc0-35a7-4fbd-a861-95fe3ca2cfb9

[含音轨 MP4](examples/videos/full4-int8-single-672124.mp4) ·
[权重来源、采样规则与校验和](examples/videos/full4-int8-single-672124.json)。
这是一例定性展示，不宣称普遍的画质等价。

VSA INT8、原生 TP4、672×384、124 帧、四步 VSA 学生模型也已完整成片：
四个 rank 均完成四次 DiT 前向，每一步检查到全部 50 个非零 gate。

https://github.com/user-attachments/assets/590be000-8cbd-458e-9198-afb2a2bdcaa4

[带音频 MP4](examples/videos/vsa-int8-tp4-672124.mp4) ·
[样例参数](examples/videos/vsa-int8-tp4-672124.json)。
逐帧检查发现该 VSA INT8 样例在约 3.4 秒处有短暂亮度回落，
同条件 BF16 在同一位置也有回落，因此不是 INT8 独有现象。
[同种子 BF16 MP4](examples/videos/vsa-bf16-tp4-672124.mp4)及
[参数记录](examples/videos/vsa-bf16-tp4-672124.json)已保留。
[时间连续性诊断](benchmarks/vsa-temporal-sample-check.json)记录了观测结果，
解码与生成 latent 的影响仍待区分。生成检查通过不代表没有闪烁或画质等价。
该样例用于定性验收，不是 Full 与 VSA 的画质等价评估。
VSA INT8 TP4 的四种视频规格现已全部通过完整生成和解码，
[审计矩阵](benchmarks/vsa-int8-tp4-matrix.json)记录各规格帧数与视频校验值。

Veda BF16、单卡昇腾 910B3、1344×768、345 帧，使用默认 LightX2V 四步 LoRA、
官方预测器和名义 90% 稀疏度，seed 42；四次 DiT 更新均输出有限值，每步经过全部 50 层。

https://github.com/user-attachments/assets/c2e08051-3753-495b-bb69-7ed8b7208b2a

[样例参数与视频校验和](examples/videos/veda4-bf16-single-1344345.json)。
已确认 345 帧和音轨，并抽样检查画面；这不构成与 Full 的画质等价证明。
此前使用 Turbo8 的 Veda INT8 TP4 四种规格验证，保留在
[历史审计矩阵](benchmarks/veda-int8-tp4-matrix.json)中。

此前严格 Spark 实验已生成 1344×768 的 124/345 帧视频，与优化前昇腾
参考实现的每个解码后 uint8 像素一致。这证明的是昇腾实现之间的一致性，
不等于与 CUDA 或 Full 输出一致，框架整合后还需要重新验收。
此前 VDN 短视频样例可正常成片。其求逆是数值近似；自动策略下的 BF16
时间卷积还会比上游 FP32 tap 累加更早发生一次特征舍入。FP32 时间卷积
选项见[架构说明](docs/architecture.md)。不宣称逐位 Cholesky 等价，
也不据单个样例宣称普遍画质等价。

首版目标是 batch=1、无参考视频的 T2VA。更多信息见
[架构设计](docs/architecture.md)、[验证状态](docs/validation.md)
及[第三方许可](THIRD_PARTY_NOTICES.md)。
