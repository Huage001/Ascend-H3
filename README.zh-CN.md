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

客户端传入与服务端相同的 `--method`：Full/VSA/Veda/VDN/Spark 均默认八步。
如需运行原版多步模型，服务端与客户端均加 `--no-few-step-lora`，客户端默认 50 步。
独立社区学生模型可显式指定 `--steps`；`--guidance-scale` 省略时保留原生策略。
本项目不额外添加 Refine。客户端保存完整 MP4 和请求/时延 JSON。

## 端到端性能

**所有方法、基线和计时请求统一为 8 次实际 DiT 前向。** Full/Veda/Spark 使用固定版本的官方 LightX2V 8-step 768p LoRA（shift 6/3）；VSA 使用官方 FastH3 V2（80% 稀疏率、shift 10/3）；VDN 保留官方八步学生。Veda 保持 90% 稀疏率。各学生保留自己的训练日程，不把四步权重强行运行八步。

加速比 = 同一硬件、同一精度、同一分辨率/帧数/种子的 **Full 八步端到端耗时 / 当前方法八步端到端耗时**。报告仅覆盖 1344×768 的 124 帧和 345 帧，单卡，BF16 / INT8；先合并 LoRA，再进行 TP 切分与量化。计时包含文本编码、全部八次去噪、音视频解码、MP4 编码和完整返回；模型加载、权重合并、预热及画面筛选不计时。

VSA 已按用户要求选定共同种子 **7**，各方法正式测速使用同一个 prompt 和种子；筛选过程已记录。下表只接受这套新口径的实测值，不把旧四步结果翻倍或换算。[此前 Full 四步为基线的数据](benchmarks/archive/pre-unified8-full4-reference.json)仅作历史归档。

<!-- BEGIN HIGHRES RESULTS -->

全部八步；共同种子：7

| 帧数 | 精度 | 方法 | 昇腾 910B3 耗时 | 昇腾加速比 | PRO 6000 耗时 | PRO 6000 加速比 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 124 | BF16 | Full + LightX2V8 | — | — | — | — |
| 124 | BF16 | VSA · FastH3 V2 · 80% | — | — | — | — |
| 124 | BF16 | Veda + LightX2V8 · 90% | — | — | — | — |
| 124 | BF16 | VDN · official | — | — | — | — |
| 124 | BF16 | Spark-H3 + LightX2V8 | — | — | — | — |
| 124 | INT8 | Full + LightX2V8 | — | — | — | — |
| 124 | INT8 | VSA · FastH3 V2 · 80% | — | — | — | — |
| 124 | INT8 | Veda + LightX2V8 · 90% | — | — | — | — |
| 124 | INT8 | VDN · official | — | — | — | — |
| 124 | INT8 | Spark-H3 + LightX2V8 | — | — | — | — |
| 345 | BF16 | Full + LightX2V8 | — | — | — | — |
| 345 | BF16 | VSA · FastH3 V2 · 80% | — | — | — | — |
| 345 | BF16 | Veda + LightX2V8 · 90% | — | — | — | — |
| 345 | BF16 | VDN · official | — | — | — | — |
| 345 | BF16 | Spark-H3 + LightX2V8 | — | — | — | — |
| 345 | INT8 | Full + LightX2V8 | — | — | — | — |
| 345 | INT8 | VSA · FastH3 V2 · 80% | — | — | — | — |
| 345 | INT8 | Veda + LightX2V8 · 90% | — | — | — | — |
| 345 | INT8 | VDN · official | — | — | — | — |
| 345 | INT8 | Spark-H3 + LightX2V8 | — | — | — | — |

统一 **Offload**：`omni-layer-prefetch1-vae-resident-v1`。加速比相对各自硬件、同精度的 Full 八步；每组对照中更高的加速比加粗，缺失或相同的值不加粗。— 为待测；旧四步或旧种子结果不进入新表。

已验收：昇腾 0/20, PRO 6000 0/20.

[八步原始记录](benchmarks/single-card-highres.json)

<!-- END HIGHRES RESULTS -->


## 效果展示与验证范围

仅展示昇腾 910B3：1344×768、345 原生帧、24 FPS（14.375 秒，约 14.4 秒），含音轨。五种方法全部八步，左列 BF16，右列原生 W8A8 INT8。共同种子固定为 **7**，其余样例正在重生成；不会把旧四步或旧种子视频改标签当成新结果。

> A cyclist rides along a quiet green park path, smooth cinematic tracking shot, natural daylight.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Full + LightX2V8 · 8 步**

新种子八步视频生成/验收中。

</td>
<td>

**Full + LightX2V8 · 8 步**

新种子八步视频生成/验收中。

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 步**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 步**

新种子八步视频生成/验收中。

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 步**

新种子八步视频生成/验收中。

</td>
<td>

**Veda + LightX2V8 · 90% · 8 步**

新种子八步视频生成/验收中。

</td>
</tr>
<tr>
<td>

**VDN · official student · 8 步**

新种子八步视频生成/验收中。

</td>
<td>

**VDN · official student · 8 步**

新种子八步视频生成/验收中。

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 步**

新种子八步视频生成/验收中。

</td>
<td>

**Spark-H3 + LightX2V8 · 8 步**

新种子八步视频生成/验收中。

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

共同种子已选定为 **7**：查看了两个候选的 16 个均匀分布帧（含末帧），seed 7 为单个骑行者，seed 123 仍为双人。seed 7 有镜头切换；不作完美连续镜头保证。筛选按用户要求进行，未按耗时挑选；该筛选请求不进入正式测速。

[新样例参数与进度](examples/videos/qualitative-1344-345.json)；[旧种子/旧配置样例归档](examples/videos/archive/qualitative-pre-unified8-seed42.json)。种子筛选按用户要求进行，筛选样例不作为计时样本，不承诺不同学生的画面逐像素或感知等价。

首版目标是 batch=1、无参考视频的 T2VA。更多信息见
[架构设计](docs/architecture.md)、[验证状态](docs/validation.md)
及[第三方许可](THIRD_PARTY_NOTICES.md)。
