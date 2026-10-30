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
| `vsa` | 官方 FastH3 V2 完整学生权重，含 50 个训练后的 compression gate | 8 步，80% 视频稀疏率；逻辑 KV64，两个选中 KV64 在片上配对为 KV128 |
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

客户端传入与服务端相同的 `--method`：Full/Veda/Spark 默认四步，VSA V2 与 VDN 八步。
如需运行原版多步模型，服务端与客户端均加 `--no-few-step-lora`，客户端默认 50 步。
独立社区学生模型可显式指定 `--steps`；`--guidance-scale` 省略时保留原生策略。
本项目不额外添加 Refine。客户端保存完整 MP4 和请求/时延 JSON。

## 端到端性能

新的单卡矩阵正在实测，仅下方“已测”数据已完成，其余仍待测。最新口径为相同硬件、精度、分辨率、
帧数下，**Full + LightX2V 四步**端到端耗时 / 对应原生学生设定端到端耗时。
量化收益另报同一方案的 BF16 / INT8 耗时比。此前 50 步基线已按要求停止，
不用于本轮图表。

模型加载、LoRA 合并、JIT 预热均在计时前完成；计时包含每次文本编码、该学生全部实际去噪前向、
音视频解码、MP4 编码及返回。Full、Veda、Spark 使用固定版本 LightX2V 4-step；
VSA 改为官方 FastH3 V2 完整八步学生（80% 稀疏率、shift 10/3）。
VSA 八步相对 Full 四步是不同配置的完整系统比较，不是同等步数的纯注意力加速比。
[旧版 VSA v1 数据](benchmarks/archive/fasth3-v1-four-step-highres.json)单独归档，不能视作 v2 结果。固定八步的 VDN 单独说明，暂不混入已完成的性能矩阵。
不同学生不宣称画质等价，也不将 50→4 步蒸馏的收益计入本轮加速比。

显存不足的设定使用 **Offload** 实测，不以 OOM 作为最终结果。PRO 6000 与昇腾
910B3 的每组对照必须使用一致的驻留、预取及编码器/解码器加载策略，并在表格和
图中注明 Offload 和策略编号。BF16、INT8 均记录实际量化模块与 BF16 回退层。
1344×768、345 帧的图将展示两种硬件量化前后的各注意力结果。
Attention/FFN 分项来自单独的性能分析运行，不能代替端到端实测。历史算子数据
保留在原始记录中，不作为这里的视频生成加速比。

加速比只报告 **1344×768、124/345 帧**两组；672×384 保留推理和功能验证支持。

<!-- BEGIN HIGHRES RESULTS -->

1344×768 · 单卡 · Full/Veda/Spark 四步、VSA V2 八步 · seed 42，**每个已测硬件/设定一次正式完整请求**：

| 帧数 | 精度 | 方法 | 昇腾 910B3 耗时 | 昇腾加速比 | PRO 6000 耗时 | PRO 6000 加速比 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 124 | BF16 | Full + LightX2V · 4 | 121.04 秒 | 1.00× | — | — |
| 124 | BF16 | VSA · FastH3 V2 · 8 · 80% | — | — | — | — |
| 124 | BF16 | Veda + LightX2V · 4 · 90% | 69.86 秒 | 1.73× | — | — |
| 124 | BF16 | Spark-H3 + LightX2V · 4 | 78.15 秒 | 1.55× | — | — |
| 124 | INT8 | Full + LightX2V · 4 | 111.90 秒 | 1.00× | — | — |
| 124 | INT8 | VSA · FastH3 V2 · 8 · 80% | — | — | — | — |
| 124 | INT8 | Veda + LightX2V · 4 · 90% | 61.78 秒 | 1.81× | — | — |
| 124 | INT8 | Spark-H3 + LightX2V · 4 | 71.41 秒 | 1.57× | — | — |
| 345 | BF16 | Full + LightX2V · 4 | 648.97 秒 | 1.00× | 250.88 秒 | 1.00× |
| 345 | BF16 | VSA · FastH3 V2 · 8 · 80% | — | — | — | — |
| 345 | BF16 | Veda + LightX2V · 4 · 90% | 225.29 秒 | **2.88×** | 103.06 秒 | 2.43× |
| 345 | BF16 | Spark-H3 + LightX2V · 4 | 276.45 秒 | **2.35×** | 115.23 秒 | 2.18× |
| 345 | INT8 | Full + LightX2V · 4 | 603.84 秒 | 1.00× | 243.86 秒 | 1.00× |
| 345 | INT8 | VSA · FastH3 V2 · 8 · 80% | — | — | — | — |
| 345 | INT8 | Veda + LightX2V · 4 · 90% | 203.77 秒 | **2.96×** | 95.43 秒 | 2.56× |
| 345 | INT8 | Spark-H3 + LightX2V · 4 | 247.17 秒 | **2.44×** | 107.27 秒 | 2.27× |

所有结果统一使用 **Offload**（`omni-layer-prefetch1-vae-resident-v1`）。加速比均相对各自硬件、同精度的 Full 四步；VSA V2 保留官方八步，因此是不同步数的端到端配置比较。加粗标记同一行中更高的加速比，相同值或尚未齐全的对照不加粗。— 表示新版待测，不能套用旧 VSA v1 的四步数字。

已完整验收：昇腾 12/16 组，PRO 6000 6/16 组。 其余组合仍在补齐，不使用预热或失败记录作为测量结果。

[高分辨率原始记录和独立分项测量](benchmarks/single-card-highres.json)。

<!-- END HIGHRES RESULTS -->

详见[端到端计时、Offload 和图表口径](docs/benchmark-protocol.md)。

## 效果展示与验证范围

仅展示昇腾 910B3 的 **1344×768、345 帧、24 FPS（14.375 秒，约 14.4 秒）**完整音视频。

左列 BF16，右列原生 W8A8 INT8；所有样例单卡、相同 prompt、seed 42、同一 Offload 策略。

Full、Veda、Spark 为四步；**VSA V2 与 VDN 保留官方八步**。VSA V2 正在重新生成，旧 VSA v1 样例已归档，不冒充 v2。

> A cyclist rides along a quiet green park path, smooth cinematic tracking shot, natural daylight.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Full + LightX2V · 4 步**

https://github.com/user-attachments/assets/cd6c4619-7cef-42fc-abc2-eebea5ed4b9b

</td>
<td>

**Full + LightX2V · 4 步**

https://github.com/user-attachments/assets/e0039d68-b50b-4f65-8714-07d4cfc3855a

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 8 步 · 80%**

新版视频验收中。

</td>
<td>

**VSA · FastH3 V2 · 8 步 · 80%**

新版视频验收中。

</td>
</tr>
<tr>
<td>

**Veda + LightX2V · 4 步**

https://github.com/user-attachments/assets/c2e08051-3753-495b-bb69-7ed8b7208b2a

</td>
<td>

**Veda + LightX2V · 4 步**

https://github.com/user-attachments/assets/ef130119-c58f-4cff-86c2-706524bbb550

</td>
</tr>
<tr>
<td>

**VDN · 官方学生 · 8 步**

https://github.com/user-attachments/assets/bc743b7a-8587-46ea-9707-a93e514befa8

</td>
<td>

**VDN · 官方学生 · 8 步**

https://github.com/user-attachments/assets/6144e35d-1a10-4d6c-91d0-55f469bae653

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V · 4 步**

https://github.com/user-attachments/assets/6ebfcaed-b023-42d5-b850-e86897c0209d

</td>
<td>

**Spark-H3 + LightX2V · 4 步**

https://github.com/user-attachments/assets/aba1d8ee-c178-4cd2-bb63-5e4a4c9f979e

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 使用 **80% 视频稀疏率**（保留预算 20%），Veda 保持 **90%**（保留预算 10%）；文本/音频前缀按原规则保持稠密，实际比例有块取整误差。

[已验收视频及待完成的 V2 样例](examples/videos/qualitative-1344-345.json)。已完成的视频核对了分辨率、345 帧、音轨及 DiT 有限输出；[旧版 VSA v1 样例来源](examples/videos/qualitative-1344-345-fasth3-v1-archive.json)单独保留。

相同 prompt 与 seed 不意味着不同学生的结果逐像素或感知等价。VDN 求逆为完整 128 维的数值近似；时间卷积的舍入差异见[架构说明](docs/architecture.md)。VSA 历史样例出现过亮度回落，见[时间连续性记录](benchmarks/vsa-temporal-sample-check.json)；此处不作无闪烁保证。

首版目标是 batch=1、无参考视频的 T2VA。更多信息见
[架构设计](docs/architecture.md)、[验证状态](docs/validation.md)
及[第三方许可](THIRD_PARTY_NOTICES.md)。
