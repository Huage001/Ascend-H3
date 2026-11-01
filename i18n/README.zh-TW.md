<p align="center">
  <img src="../docs/assets/ascend-h3-logo.png" alt="Ascend-H3 logo" width="190">
</p>

<h1 align="center">Ascend-H3</h1>

<div align="center">

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-orange)](../LICENSE)
[![Python: 3.12](https://img.shields.io/badge/python-3.12-blue)](../docs/environment.md)
[![OS: Linux](https://img.shields.io/badge/OS-Linux-lightgrey)](../docs/environment.md)
[![Platform: Ascend 910B3](https://img.shields.io/badge/platform-Ascend%20910B3-0077C8)](../docs/validation.md)
[![Backend: vLLM-Omni](https://img.shields.io/badge/backend-vLLM--Omni-orange)](../docs/environment.md)
[![Email: liusonghua@sjtu.edu.cn](https://img.shields.io/badge/%E2%9C%89-Email-0077C8)](mailto:liusonghua@sjtu.edu.cn)

</div>

<p align="center"><a href="#install">Installation</a> | <a href="#models">Models</a> | <a href="#serve">Serve</a> | <a href="#performance">Performance</a></p>

<div align="center">

[English](../README.md) | [简体中文](../README.zh-CN.md) | **繁體中文** | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> **License:** Project-authored contributions in this distribution are offered under the [PolyForm Noncommercial License 1.0.0](../LICENSE). Third-party code and model weights have separate terms, and earlier Apache-2.0 grants remain in effect. See [third-party notices](../THIRD_PARTY_NOTICES.md). For uses not covered by the applicable terms, contact [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn).

> 這份完整的自述文件是由人工智慧輔助翻譯的英文自述文件（提交為 be5c42e03f9a）。它可能會變得過時；請參閱英文版以獲取最新資訊。 [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

MiniMax 對 Ascend 的 H3 推理，在一個存儲庫中具有完整、VSA、Veda、VDN 和 Spark-H3 關注。此整合重用了 vLLM-Omni 模型載入、INT8 轉換器量化和本機張量並行注意頭。

**驗證範圍：** 40/40 INT8 E2E 單元，1344×768、124/345 幀和 1/2/4/8 NPU（37 個新測量值和 3 個明確標記的歷史層卸載單元）。 Ascend 910B3 和 PRO 6000 上的駐留 BF16 比較包含 20 個注意力核心和 20 個完整區塊測量。注意力算子保留其經過驗證的版本。

## 預設 768p 工作負載

| 案例 | 寬×高 | 本機輸出幀 | 潛伏視訊網格 |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

輸出 FPS 為 24。請求使用顯式幀計數，並停用幀插值。長的情況是商定的本機 345 幀工作負載；它沒有被裁剪為 341 幀。 BF16 和 INT8 使用這兩個 1344×768 工作負載。省略 `--case` 選擇 `1344-124`。

<a id="install"></a>

## 安裝

使用Linux、Python 3.12和相容的Ascend Torch/CANN環境。安裝[環境說明](../docs/environment.md) 中所述的固定 vLLM、vLLM-Ascend 和 vLLM-Omni 堆疊。不要合併不相關的框架版本。然後從來源安裝此儲存庫：

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

建置不包含模型權重，執行時間不佔用 NPU。 本機庫在本地編譯並保存在結帳之外。 對於單一後端，將 `all` 替換為 `vsa`、`veda` 或 `spark`； Full 和 VDN 不需要這些自訂 CANN 擴充。 保持可編輯的結帳可用。

<a id="models"></a>

## 檢查點和演算法選擇

| 方法 | 所需的模型資產 | 計算 |
| --- | --- | --- |
| `full` | H3底座；LightX2V 預設8步，或社群 LoRA | 密集關注 |
| `vsa` | 完整的官方 FastH3 V2 學生，包括所有 50 個經過訓練的壓縮門 | 8個轉發，80%視訊稀疏度；邏輯 KV64，選定的 KV 區塊與片上 KV128 配對 |
| `veda` | H3基礎+LightX2V 8步+官方訓練的Veda預測器 | 動態每頭 Veda 區塊路線，名目稀疏度 90% |
| `vdn` | H3基礎+官方VDN階段-DMD檢查點目錄 | 局部softmax加上128維學習的DeltaNet分支 |
| `spark` | H3底座+LightX2V 8步 | Spark-H3 分層分區和聯合精確/代理 Softmax，標稱 90% 精確分支稀疏度 |

根據原始許可證下載模型。權重不會重新分配。不同的蒸餾生有不同的時間表和學習規則；將注意力轉移到任意權重上並不能保證匹配品質。 VSA 需要非零訓練門。 VDN 需要其學習的分支權重。下載和準備指令位於[模型資產](../docs/weights.md)中。

Full、Veda 和 Spark 預設啟用固定 **LightX2V 八步 LoRA**。使用 `--no-few-step-lora` 運行原始多步驟模型。準備好的社區`--lora /models/style`在加載完成前與LightX2V融合在一起；明確選擇另一個社區學生時禁用預設學生。 Veda 始終需要其經過官方訓練的預測器。 VSA僅接受完整的官方**FastH3 V2完整檢查點**：八個轉發，80％視頻稀疏性，視頻/音頻移位10/3，tile64，指導1。它直接載入發布的權重，不使用`--lora`。 VDN 只接受其完整的官方階段-DMD 檢查點。兩者都不接受額外的社區 LoRA 或學生禁用標誌。

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

一旦使用其聲明的 alpha/rank，官方 PEFT 來源就會被標準化；準備好的文件經過哈希檢查。所有選定的適配器在 TP 切片和 INT8 量化之前合併。沒有請求時 LoRA 矩陣乘法。有關來源格式和固定時間表，請參閱[模型準備](../docs/weights.md)。

<a id="serve"></a>

## 啟動伺服器

僅使用沒有其他計算進程的設備。對於一張卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

其他後端使用相同的伺服器 API：

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

在選定的設備上一次運行一台伺服器。新增 `--dry-run` 以檢查產生的本機 vLLM-Omni 命令，而無需載入模型。

單卡範例串流文字編碼器，同時保持 INT8 DiT 駐留。本機配方量化DiT；文字編碼器仍為 BF16，因此其記憶體佔用仍需要包含在啟動策略中。對於 **BF16**，使用 `--precision bf16`；如有必要，明確選擇 `--offload layer --no-precompute-adaln`。 INT8 使用原生配方：

```json
{"transformer":{"method":"int8"}}
```

INT8 投影將 BF16 Q/K/V 引起注意。注意力累加器、Veda 預測和 VDN 狀態保持其規定的精確度。這是INT8模型推理，而不是聲稱每個注意力操作都是INT8。本機配方保留 BF16 中不受支援的層。單卡H3具有96,768寬的投影，超過了NPU量化matmul輸出限制65,535；TP 可以將其每列寬度縮小到範圍內。

對於**四卡推理**，使用原生張量並行頭分片：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices`支援1、2、4、7、8，共分割56個頭。 Veda 分片後保留全域預測器頭 ID。七卡DiT TP使用一個文字編碼器等級，因為它的64個查詢/8個KV頭不能被七除。預設啟動為 `--offload none`（常駐權重）。如果記憶體不足，`--offload text` 僅傳輸文字編碼器，同時將 DiT 保留在裝置上；`--offload module` 更換完整元件（實驗合格）；`--offload layer` 傳輸兩個元件，保持一個活動區塊和一個預取區塊。以下完成的匹配卸載參考在兩個硬體平台上明確使用 `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`)。新的端到端測量僅使用**INT8**，從常駐執行開始並標記每個必要的卸載回退。隔離注意力和 DiT 區塊比較在兩個裝置上使用 BF16：層 0/24/49，每個運行 3 次，對 9 個時序求和並除以 9。可選的 `--parallel ulysses` 暴露給非 VDN 路徑； TP 保留預設值。本機平鋪 VAE 解碼現在預設為所有選定的卡片 (`--vae-parallel-size 0`)；將 `--vae-parallel-size 1` 用於單一解碼器等級。 INT8 全/124 幀管道已通過一、二、四和八 VAE 等級。其他方法/幀計數組合保留在 E2E 矩陣中。

駐留 NPU 在伺服器準備就緒之前啟動準備 AdaLN 結果。準備解析載入的模型並合併LoRA的實際西格瑪位置、DiT更新計數和視訊/音訊移位；它沒有被硬編碼為八個步驟。使用 `serve --steps N` 指定教師/自訂學生啟動，並在 `generate --steps N` 中使用相同的計數。衝突的精簡檢查點計畫被拒絕。快取容量隨著所需的計劃而成長。保留原始投影數學； AdaLN 主權重保留在 CPU 上，而準備好的結果保留在 NPU 上。 TP4/TP8 還可以記憶來自精確時間步輸入的本機時間嵌入，以避免低位 FP32 減少變化使下游快取無效。權重、數值設定和全等級命中協議仍保證重用。使用新配置重新啟動以預先計算不同的時間表；未準備的輸入使用原始計算並可能傳輸權重。 `--no-precompute-adaln` 停用準備；DiT 模組/層流預設保持停用狀態。

對於單卡345幀Spark/VSA/VDN，使用`--offload layer --no-precompute-adaln`和原始驗證算子。 E2E 表格以 †L 標示這些本機層 Offload 單元； ‡R 標識上次執行時重複使用的符合測量值。所有其他單位保留其單獨記錄的居住政策。

啟動器將本機工作執行緒請求截止時間設為 7200 秒，因此長影片不會因上游 600 秒預設值而中止。在啟動前設定 `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` 以覆蓋此限制；它改變等待行為，而不是模型計算或測量的請求持續時間。

## 產生兩個 768p 案例

建立一個 UTF-8 提示文件，啟動伺服器，然後使用其匹配的步數：

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

將伺服器的`--method`傳遞給客戶端：所有五種方法（Full/VSA/Veda/VDN/Spark）預設為八個實際轉送。對於可選的原始多步驟模型，使用`generate --method full --no-few-step-lora`（50步），與伺服器相符。明確的 `--steps` 仍然可供單獨準備的社區學生使用。 `--guidance-scale`為可選；遺漏保留了本土政策。沒有添加Refine階段。客戶端保存 MP4 和 JSON 計時收據。

<!-- BEGIN BF16 LAYER RESULTS -->

## BF16 層比較：駐留在兩個裝置上

僅限 1344×768、124/345 幀。真實的學生權重和位元組相同的凍結合併成隱藏狀態；這些不是從視訊請求捕獲的層激活。第 0、24 和 49 層各接收兩次不計時暖身和三次正式運行：**將九個計時相加並除以九**。權重保留在兩個設備上。啟動/主要權重為 BF16；保留所需的 FP32 預測器、累積和 VDN 狀態。加速使用相同硬體、相同形狀的 BF16 完整基準。較高的相對加速比是粗體；也顯示了絕對延遲。

### 僅注意核心

從準備好的 Q/K/V 開始，到注意力輸出結束。不包括 QKV 投影、Q/K 範數、RoPE、輸出投影、AdaLN 和 FFN。包括所需的稀疏閘、預測、選擇、重排和恢復；VDN 包括完整的 softmax 和線性核心。每個方法都使用這個邊界；整個模組的計時不會被取代。核心時序已完成10/10匹配的硬體對；待處理的單元不會重複使用早期的模組測量。

| 鏡框 | 方法 | Ascend 延遲 | PRO 6000 延遲 | Ascend 加速 | PRO 6000 加速 |
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

### 完整的 DiT 塊

從區塊輸入運行到區塊輸出，包括投影、歸一化、RoPE、注意力核心、AdaLN、FFN和殘差操作。獨立於磁芯進行測量；它們的差異不是 FFN 測量，也不會外推到 E2E 時間。

| 鏡框 | 方法 | Ascend 延遲 | PRO 6000 延遲 | Ascend 加速 | PRO 6000 加速 |
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

![BF16關注核心與DiT區塊對比](../docs/figures/bf16-three-layer.png)[所有九輪記錄和輸入身份檢查](../benchmarks/bf16-three-layer.json)。

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E：Ascend 1 / 2 / 4 / 8 卡

1344×768，DiT 八個實際更新，包括文字編碼、視訊/音訊解碼和 MP4 寫入。不包括載入、先前的 LoRA 合併、AdaLN 準備和預熱。優先選擇居民DiT；常駐行驗證零 AdaLN 未命中以及圍繞正式請求的每個等級的權重轉移。記錄OOM後，標記的單卡345幀單元使用本機層卸載和重用匹配，根據要求充分驗證先前的測量；這些單元格被明確標記，而不是作為新的測量值呈現。

| 鏡框 | 方法 | 1 張卡 | 2張卡 | 4張卡 | 8張卡 |
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

覆蓋 **40/40**：37 個新鮮的和 3 個重複使用的先前測量值。 †T：仅文本编码器层卸载； †L：本機 DiT/文字編碼器層卸載，具有一層預取和駐留 VAE； ‡R：符合 2026-10-31 的原始測量值。 未標記的單元沒有運行時卸載。 仅驻留行预计算 AdaLN； 層卸載行不提出 DiT 駐留或零 AdaLN 轉移聲明。 VAE 對比賽牌數進行排名； DiT 頭分體 56/28/14/7。 保留本機 INT8 覆蓋範圍。[INT8 E2E記錄](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

早期配對的layer-Offload對比保留在[歷史單卡參考](../docs/archive/matched-offload-single-card.md)中。當前表中重複使用的儲存格連結到未變更的原始記錄。

## 定性結果和驗證

僅限 Ascend 910B3：**1344×768，345 原生幀，24 FPS（14.375 秒，約 14.4 秒）**，有音訊。

所有五種方法都使用八個轉發。左：BF16。右：原生W8A8 INT8。所有樣本都使用一張卡、相同的提示、通用種子 **7** 和匹配的卸載。

> 騎自行車的人沿著安靜的綠色公園小徑騎行，流暢的電影跟踪鏡頭和自然光。

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**完整+LightX2V8·8步**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**完整+LightX2V8·8步**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 個步驟**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 個步驟**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 個步驟**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 個步驟**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN·官方·8步**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN·官方·8步**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 個步驟**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 個步驟**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2採用80%視訊稀疏度；Veda 使用 90%。文字和音訊保留其固有規則。[所有十個影片的參數、SHA256 和來源](../examples/videos/qualitative-1344-345.json)。全解碼驗證345幀和音訊；每個剪輯有 16 個統一的幀進行視覺審查。不同的學生並不被認為在像素或感知上是等效的。 VSA 保留鏡頭變化；不保證無閃爍。

初始整合目標是第一批沒有參考影片的 T2VA。請參閱[架構](../docs/architecture.md)、[驗證狀態](../docs/validation.md)和[第三方聲明](../THIRD_PARTY_NOTICES.md)。
