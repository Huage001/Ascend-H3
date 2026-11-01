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

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | **日本語** | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> **License:** Project-authored contributions in this distribution are offered under the [PolyForm Noncommercial License 1.0.0](../LICENSE). Third-party code and model weights have separate terms, and earlier Apache-2.0 grants remain in effect. See [third-party notices](../THIRD_PARTY_NOTICES.md). For uses not covered by the applicable terms, contact [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn).

> この完全な README は、コミット be5c42e03f9a にある英語の README の AI 支援による翻訳です。それは時代遅れになる可能性があります。最新情報については英語版を参照してください。 [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

MiniMax 1 つのリポジトリでフル、VSA、Veda、VDN、および Spark-H3 アテンションを使用した Ascend の H3 推論。この統合では、vLLM-Omni モデルの読み込み、INT8 トランスフォーマーの量子化、およびネイティブ テンソル並列アテンション ヘッドを再利用します。

**検証範囲:** 1344×768、124/345 フレームおよび 1/2/4/8 NPU の 40/40 INT8 E2E セル (37 個の新しい測定値と 3 個の明示的にマークされた履歴レイヤーオフロード セル)。 Ascend 910B3 と PRO 6000 の常駐 BF16 比較には、20 個のアテンション コア測定値と 20 個の完全ブロック測定値が含まれています。アテンション演算子は検証されたバージョンを保持します。

## デフォルトの 768p ワークロード

| ケース | 幅×高さ | ネイティブ出力フレーム | 潜在ビデオグリッド |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

出力 FPS は 24 です。リクエストでは明示的なフレーム カウントが使用され、フレーム補間は無効になります。長いケースは、合意されたネイティブ 345 フレーム ワークロードです。 341 フレームにはクロップされません。 BF16 と INT8 は、これら 2 つの 1344×768 ワークロードを使用します。 `--case` を省略すると、`1344-124` が選択されます。

<a id="install"></a>

## インストール

Linux、Python 3.12、および互換性のある Ascend トーチ/CANN 環境を使用します。[環境に関する注意事項](../docs/environment.md) で説明されている固定された vLLM、vLLM-Ascend および vLLM-Omni スタックをインストールします。無関係なフレームワーク リリースを組み合わせないでください。次に、このリポジトリをソースからインストールします。

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

ビルドにはモデルの重みが含まれず、NPU を占有せずに実行されます。ネイティブ ライブラリはローカルでコンパイルされ、チェックアウトの外部に保持されます。単一のバックエンドの場合、`all` を `vsa`、`veda`、または `spark` に置き換えます。フルおよび VDN には、これらのカスタム CANN 拡張機能は必要ありません。編集可能なチェックアウトを利用可能な状態に保ちます。

<a id="models"></a>

## チェックポイントとアルゴリズムの選択

| 方法 | 必要なモデルアセット | 計算 |
| --- | --- | --- |
| `full` | H3ベース。 LightX2V デフォルトで 8 ステップ、またはコミュニティ LoRA | 集中的な注意 |
| `vsa` | 50 個のトレーニング済み圧縮ゲートをすべて含む完全な公式 FastH3 V2 スチューデント | 8 つの転送、80% のビデオ スパース性。論理 KV64、選択された KV ブロックがチップ上の KV128 にペアリングされる |
| `veda` | H3 ベース + LightX2V 8 ステップ + 公式トレーニング済み Veda 予測子 | ヘッドごとの動的 Veda ブロック ルート、公称 90% のスパース性 |
| `vdn` | H3 ベース + 公式 VDN ステージ - DMD チェックポイント ディレクトリ | ローカル ソフトマックスと 128 次元の学習済み DeltaNet ブランチ |
| `spark` | H3ベース+LightX2V 8段 | Spark-H3 階層パーティションとジョイント完全/プロキシ ソフトマックス、公称 90% 完全ブランチ スパース性 |

元のライセンスに基づいてモデルをダウンロードします。重みは再分配されません。蒸留された生徒が異なれば、スケジュールも学習したルールも異なります。任意の重みに対する注意を交換しても、一致の品質は保証されません。 VSA には非ゼロのトレーニング済みゲートが必要です。 VDN には、学習されたブランチの重みが必要です。ダウンロードと準備のコマンドは[モデル アセット](../docs/weights.md) にあります。

Full、Veda、および Spark では、固定された **LightX2V 8 ステップ LoRA** がデフォルトで有効になります。 `--no-few-step-lora` を使用して、元の多ステップ モデルを実行します。準備されたコミュニティ `--lora /models/style` は、ロードが完了する前に LightX2V と融合されます。別のコミュニティの生徒を明示的に選択するときに、デフォルトの生徒を無効にします。 Veda には、公式にトレーニングされた予測子が常に必要です。 VSA は、完全な公式 **FastH3 V2 完全チェックポイント** のみを受け入れます: 8 つのフォワード、80% のビデオ スパース性、ビデオ/オーディオ シフト 10/3、タイル 64、ガイダンス 1。リリースされた重みを直接ロードし、`--lora` を受け取りません。 VDN は、完全な公式ステージ DMD チェックポイントのみを受け入れます。どちらも追加のコミュニティ LoRA や学生無効化フラグを受け入れません。

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

公式 PEFT ソースは、宣言されたアルファ/ランクを使用して一度正規化されます。準備されたファイルはハッシュチェックされます。選択されたすべてのアダプターは、TP スライスと INT8 量子化の前にマージされます。リクエスト時の LoRA 行列の乗算はありません。ソース形式と固定スケジュールについては、[モデルの準備](../docs/weights.md) を参照してください。

<a id="serve"></a>

## サーバーを起動する

他のコンピューティング プロセスを持たないデバイスのみを使用してください。 1 枚のカードの場合:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

他のバックエンドは同じサーバー API を使用します。

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

選択したデバイス上で一度に 1 つのサーバーを実行します。モデルをロードせずに、生成されたネイティブ vLLM-Omni コマンドを検査するには、`--dry-run` を追加します。

シングルカードの例では、INT8 DiT を常駐させたまま、テキスト エンコーダーをストリーミングします。ネイティブ レシピは DiT を量子化します。テキスト エンコーダーは BF16 のままであるため、そのメモリ フットプリントを起動ポリシーに含める必要があります。 **BF16** の場合は、`--precision bf16` を使用します。必要に応じて、`--offload layer --no-precompute-adaln` を明示的に選択します。 INT8 はネイティブ レシピを使用します。

```json
{"transformer":{"method":"int8"}}
```

INT8 プロジェクションは、BF16 Q/K/V に注目を集めます。アテンションアキュムレータ、Veda 予測、および VDN 状態は、規定の精度を維持します。これは INT8 モデルの推論であり、すべてのアテンション操作が INT8 であるという主張ではありません。ネイティブ レシピは、サポートされていないレイヤーを BF16 に保持します。シングルカード H3 には、NPU 量子化された matmul 出力制限の 65,535 を超える 96,768 幅のプロジェクションがあります。 TP はランクごとの幅を範囲内に縮小できます。

**4 カード推論**の場合は、ネイティブ テンソル並列ヘッド シャーディングを使用します。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices` は 56 ヘッドに分割する 1、2、4、7、8 をサポートします。 Veda は、シャーディング後にグローバル プレディクター ヘッド ID を保持します。 7 枚のカード DiT TP は、64 クエリ / 8 KV ヘッドを 7 で割ることができないため、1 つのテキスト エンコーダ ランクを使用します。デフォルトで `--offload none` (常駐重み) を起動します。メモリが不十分な場合、`--offload text` はデバイス上に DiT を保持しながら、テキスト エンコーダーのみをストリーミングします。 `--offload module` は完全なコンポーネントを交換します (実験的認定)。 `--offload layer` は両方のコンポーネントをストリーミングし、1 つのアクティブなブロックと 1 つのプリフェッチされたブロックを維持します。以下の完成した matched-Offload リファレンスでは、両方のハードウェア プラットフォームで `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) を明示的に使用しています。新しいエンドツーエンド測定では **INT8 のみ**を使用し、常駐実行から開始して、必要なすべてのオフロード フォールバックをマークします。分離されたアテンションと DiT ブロックの比較では、両方のデバイスで BF16 を使用します: レイヤー 0/24/49、それぞれ 3 回実行し、9 つのタイミングを合計して 9 で除算します。オプションの `--parallel ulysses` は、非 VDN パスに対して公開されます。 TP はデフォルトのままです。ネイティブ タイル VAE デコードは、選択されたすべてのカード (`--vae-parallel-size 0`) にデフォルトで適用されるようになりました。単一のデコーダ ランクには `--vae-parallel-size 1` を使用します。 INT8 フル/124 フレーム パイプラインは、VAE ランク 1、2、4、および 8 で合格しました。他の方式とフレーム数の組み合わせは E2E マトリックスに残ります。

常駐 NPU は、サーバーの準備が整う前に AdaLN の結果を準備します。準備では、ロードされたモデルを解決し、LoRA の実際のシグマ位置、DiT 更新数、およびビデオ/オーディオ シフトをマージします。 8 つのステップにハードコードされているわけではありません。 `serve --steps N` を使用して教師/カスタム生徒の起動を指定し、`generate --steps N` で同じカウントを使用します。競合する抽出されたチェックポイント スケジュールは拒否されます。キャッシュ容量は、必要なプランに応じて増加します。元の投影計算は保持されます。 AdaLN マスター ウェイトは CPU に残りますが、準備された結果は NPU に残ります。 TP4/TP8 はまた、ダウンストリーム キャッシュを無効にする低ビット FP32 リダクション変動を回避するために、正確なタイムステップ入力からのネイティブ時間埋め込みをメモ化します。重み、数値設定、および全ランクのヒット合意は依然としてガードの再利用を保証します。新しい構成で再起動して、別のスケジュールを事前計算します。準備されていない入力では元の計算が使用され、重みが転送される可能性があります。 `--no-precompute-adaln` は準備を無効にします。 DiT モジュール/レイヤー ストリーミングは、デフォルトで無効のままになります。

シングルカード 345 フレーム Spark/VSA/VDN の場合は、元の検証済み演算子で `--offload layer --no-precompute-adaln` を使用します。 E2E テーブルでは、これらのネイティブ レイヤー オフロード セルに †L のマークが付いています。 ‡R は、前回の実行から再利用された一致する測定値を識別します。他のすべてのセルは、個別に記録された居住ポリシーを保持します。

ランチャーはネイティブ ワーカー リクエストの期限を 7200 秒に設定するため、長いビデオがアップストリームのデフォルトの 600 秒によって中止されません。この制限をオーバーライドするには、起動前に `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` を設定します。モデルの計算や測定されたリクエスト期間ではなく、待機動作が変更されます。

## 2 つの 768p ケースを生成する

UTF-8 プロンプト ファイルを作成し、サーバーを起動して、一致するステップ数を使用します。

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

サーバーの `--method` をクライアントに渡します。5 つのメソッド (Full/VSA/Veda/VDN/Spark) はすべて、デフォルトで 8 つの実際の転送に設定されます。オプションのオリジナル多段モデルの場合は、サーバーに合わせて`generate --method full --no-few-step-lora`(50段)をご使用ください。別途用意されたコミュニティの学生は、明示的な `--steps` を引き続き利用できます。 `--guidance-scale` はオプションです。省略するとネイティブ ポリシーが保持されます。リファインステージは追加されません。クライアントは MP4 と JSON タイミング受信を保存します。

<!-- BEGIN BF16 LAYER RESULTS -->

## BF16 レイヤーの比較: 両方のデバイスに常駐

1344×768、124/345フレームのみ。実際のスチューデントの重みとバイト同一の凍結された合成隠れ状態。これらはビデオリクエストからキャプチャされたレイヤーアクティベーションではありません。レイヤー 0、24、および 49 はそれぞれ、2 つの時間なしウォームアップと 3 つの正式な実行を受け取ります。**9 つのタイミングを合計して 9 で割ります**。重みは両方のデバイスに常駐します。アクティベーション/メインの重みは BF16 です。必要な FP32 予測子、累積値、および VDN 状態は保持されます。高速化には、同じハードウェア、同じ形状の BF16 完全なベースラインが使用されます。相対的な高速化は大胆です。絶対的なレイテンシーも表示されます。

### アテンションコアのみ

準備された Q/K/V で始まり、アテンション出力で終了します。 QKV 投影、Q/K ノルム、RoPE、出力投影、AdaLN、FFN は除きます。必要なスパース ゲート、予測、選択、再配置、復元が含まれます。 VDN には、完全なソフトマックス コアとリニア コアの両方が含まれています。すべてのメソッドはこの境界を使用します。モジュール全体のタイミングは置き換えられません。コア タイミングは 10/10 の一致したハードウェア ペアを完了しました。保留中のセルは、以前のモジュール測定値を再利用しません。

| フレーム | 方法 | Ascend レイテンシ | PRO 6000 の遅延 | Ascend の高速化 | PRO 6000 の高速化 |
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

### DiT ブロックを完了する

投影、正規化、RoPE、アテンション コア、AdaLN、FFN、残差演算など、ブロック入力からブロック出力まで実行されます。コアとは独立して測定されます。それらの差は FFN 測定値ではなく、どちらも E2E 時間に外挿されません。

| フレーム | 方法 | Ascend レイテンシ | PRO 6000 の遅延 | Ascend の高速化 | PRO 6000 の高速化 |
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

![BF16 アテンションコアと DiT ブロックの比較](../docs/figures/bf16-three-layer.png)[9 回の実行レコードすべてと入力 ID チェック](../benchmarks/bf16-three-layer.json)。

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1 / 2 / 4 / 8 カード

1344×768、テキスト エンコード、ビデオ/オーディオ デコード、MP4 書き込みを含む 8 つの実際の DiT アップデート。ロード、以前の LoRA マージ、AdaLN の準備およびウォームアップは除外されます。常駐 DiT を優先します。常駐行は、正式なリクエストの周囲のすべてのランクで AdaLN ミスと重み付け転送がゼロであることを検証します。 OOM が記録された後、マークされたシングルカード 345 フレーム セルはネイティブ レイヤ オフロードを使用し、要求に応じてマッチング、完全に検証された以前の測定を再利用します。これらのセルは、新しい測定値として表示されるのではなく、明示的にマークされます。

| フレーム | 方法 | カード1枚 | カード2枚 | カード4枚 | カード8枚 |
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

**40/40** をカバー: 37 件の新鮮な測定値と 3 件の再利用された以前の測定値。 †T: テキストエンコーダ層オフロードのみ。 †L: ネイティブ DiT/テキスト エンコーダー層 1 層プリフェッチと常駐 VAE によるオフロード。 ‡R: 2026-10-31 の元の測定値と一致します。マークのないセルにはランタイム オフロードがありません。常駐行のみが AdaLN を事前計算します。レイヤーオフロード行では、DiT 常駐またはゼロ AdaLN 転送要求は行われません。 VAE はカード数と一致するようにランク付けします。 DiT ヘッドは 56/28/14/7 に分割されています。ネイティブの INT8 カバレッジは保持されます。[INT8 E2E レコード](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

以前に一致したレイヤーとオフロードの比較は、[歴史的なシングルカード リファレンス](../docs/archive/matched-offload-single-card.md) に保持されます。現在のテーブルで再利用されたセルは、変更されていない元のレコードにリンクされます。

## 定性的な結果と検証

Ascend 910B3 のみ: **1344×768、345 ネイティブ フレーム、24 FPS (14.375 秒、約 14.4 秒)**、オーディオ付き。

5 つのメソッドはすべて 8 つのフォワードを使用します。左: BF16。右: ネイティブ W8A8 INT8。すべてのサンプルは 1 つのカード、同じプロンプト、共通シード **7**、および一致するオフロードを使用します。

> サイクリストが静かな緑豊かな公園の小道、スムーズな映画のようなトラッキング ショット、自然光に沿って走行します。

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**フル + LightX2V8 · 8 ステップ**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**フル + LightX2V8 · 8 ステップ**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 ステップ**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 ステップ**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 ステップ**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 ステップ**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · 公式 · 8 ステップ**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · 公式 · 8 ステップ**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 ステップ**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 ステップ**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 は 80% のビデオ スパース性を使用します。 Veda は 90% を使用します。テキストとオーディオはネイティブのルールを保持します。[10 本のビデオすべてのパラメータ、SHA256、および出所](../examples/videos/qualitative-1344-345.json)。完全なデコードでは 345 フレームとオーディオを検証します。クリップごとに 16 個の均一なフレームを視覚的にレビューしました。異なる生徒がピクセル上または知覚的に同等であるとは主張されません。 VSA はショットの変更を保持します。ちらつきのない保証はありません。

初期統合は、参照ビデオなしのバッチ 1 T2VA をターゲットとしています。[アーキテクチャ](../docs/architecture.md)、[検証ステータス](../docs/validation.md)、および[サードパーティ通知](../THIRD_PARTY_NOTICES.md)を参照してください。
