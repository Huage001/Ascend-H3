<p align="center">
  <img src="../docs/assets/ascend-h3-logo.png" alt="Ascend-H3 logo" width="190">
</p>

<h1 align="center">Ascend-H3</h1>

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](../LICENSE)
[![Python: 3.12](https://img.shields.io/badge/python-3.12-blue)](../docs/environment.md)
[![OS: Linux](https://img.shields.io/badge/OS-Linux-lightgrey)](../docs/environment.md)
[![Platform: Ascend 910B3](https://img.shields.io/badge/platform-Ascend%20910B3-0077C8)](../docs/validation.md)
[![Backend: vLLM-Omni](https://img.shields.io/badge/backend-vLLM--Omni-orange)](../docs/environment.md)

</div>

<p align="center"><a href="#install">Installation</a> | <a href="#models">Models</a> | <a href="#serve">Serve</a> | <a href="#performance">Performance</a></p>

<div align="center">

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | **Türkçe**

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> Bu tam README, be5c42e03f9a taahhüt adresindeki İngilizce README'nin yapay zeka destekli bir çevirisidir. Güncelliğini yitirebilir; En son bilgiler için İngilizce versiyona bakın. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

Tam, VSA, Veda, VDN ve Spark-H3 dikkatine sahip Ascend üzerinde MiniMax H3 çıkarımı tek bir depoda. Entegrasyon, vLLM-Omni model yüklemesini, INT8 transformatör kuantizasyonunu ve yerel tensör paralel dikkat kafalarını yeniden kullanır.

**Doğrulanmış kapsam:** 1344×768, 124/345 kare ve 1/2/4/8 NPU'larda 40/40 INT8 E2E hücresi (37 yeni ölçüm ve açıkça işaretlenmiş 3 tarihsel katman-Boşaltma hücresi). Ascend 910B3 ve PRO 6000'deki yerleşik BF16 karşılaştırması, 20 dikkat merkezi ve 20 tam blok ölçümü içerir. Dikkat operatörleri doğrulanmış sürümlerini korur.

## Varsayılan 768p iş yükleri

| Vaka | Genişlik × yükseklik | Yerel çıktı çerçeveleri | Gizli video ızgarası |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

Çıkış FPS'si 24'tür. İstekler, çerçeve enterpolasyonu devre dışı bırakılarak açık çerçeve sayıları kullanır. Uzun durum, üzerinde anlaşmaya varılan yerel 345 çerçeveli iş yüküdür; 341 kareye kırpılmaz. BF16 ve INT8 bu iki 1344×768 iş yükünü kullanır. `--case`'nin atlanması `1344-124`'yi seçer.

<a id="install"></a>

## Kurulum

Linux, Python 3.12 ve uyumlu bir Ascend Torch/CANN ortamını kullanın.[Ortam notları](../docs/environment.md) bölümünde açıklanan sabitlenmiş vLLM, vLLM-Ascend ve vLLM-Omni yığınını yükleyin. İlgisiz çerçeve sürümlerini birleştirmeyin. Ardından bu depoyu kaynaktan yükleyin:

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

Yapılar model ağırlığı içermez ve NPU'yu işgal etmeden çalışır. Yerel kütüphaneler yerel olarak derlenir ve ödemenin dışında tutulur. Tek bir arka uç için `all`'yi `vsa`, `veda` veya `spark` ile değiştirin; Full ve VDN, bu özel CANN uzantılarını gerektirmez. Düzenlenebilir ödemeyi kullanılabilir durumda tutun.

<a id="models"></a>

## Kontrol noktaları ve algoritma seçenekleri

| Yöntem | Gerekli model varlıkları | hesaplama |
| --- | --- | --- |
| `full` | H3 bazı; LightX2V Varsayılan olarak 8 adımlı veya topluluk LoRA | Yoğun dikkat |
| `vsa` | Eğitimli 50 sıkıştırma kapısının tümü dahil olmak üzere eksiksiz resmi FastH3 V2 öğrencisi | 8 ileri, %80 video seyrekliği; mantıksal KV64, seçilen KV blokları çip üzerinde KV128 ile eşleştirildi |
| `veda` | H3 tabanı + LightX2V 8 adımlı + resmi eğitimli Veda tahmin aracı | Dinamik kafa başına Veda blok rotaları, nominal %90 seyreklik |
| `vdn` | H3 tabanı + resmi VDN stage-DMD kontrol noktası dizini | Yerel softmax artı 128 boyutlu öğrenilmiş DeltaNet şubesi |
| `spark` | H3 tabanı + LightX2V 8 adımlı | Spark-H3 hiyerarşik bölüm ve ortak tam/proxy softmax, nominal %90 tam dal seyrekliği |

Modelleri orijinal lisansları altında indirin. Ağırlıklar yeniden dağıtılmaz. Farklı damıtılmış öğrencilerin farklı programları ve öğrenilmiş kuralları vardır; Dikkatin keyfi ağırlıklara kaydırılması eşleştirme kalitesini garanti etmez. VSA sıfır olmayan eğitimli geçitler gerektirir. VDN öğrenilen dal ağırlıklarını gerektirir. İndirme ve hazırlama komutları[model varlıkları](../docs/weights.md) içindedir.

Full, Veda ve Spark, sabitlenmiş **LightX2V sekiz adımlı LoRA**'yi varsayılan olarak etkinleştirir. Orijinal çok adımlı modeli çalıştırmak için `--no-few-step-lora`'yi kullanın. Hazırlanan bir topluluk `--lora /models/style`, yükleme tamamlanmadan önce LightX2V ile birleştirilir; Açıkça başka bir topluluk öğrencisini seçerken varsayılan öğrenciyi devre dışı bırakın. Veda her zaman resmi eğitimli tahmincisine ihtiyaç duyar. VSA yalnızca resmi **FastH3 V2 tam kontrol noktasının** tamamını kabul eder: sekiz ileri, %80 video seyrekliği, video/ses 10/3 kaydırma, döşeme64, kılavuz 1. Serbest bırakılan ağırlıkları doğrudan yükler ve `--lora`'yi almaz. VDN yalnızca resmi aşama DMD kontrol noktasının tamamını kabul eder. Hiçbiri ek bir topluluk LoRA veya öğrenci engelli bayrağını kabul etmez.

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

Resmi PEFT kaynağı, beyan edilen alfa/derecesi kullanılarak bir kez normalleştirilir; Hazırlanan dosyalar hash kontrolünden geçirilir. Seçilen tüm bağdaştırıcılar, TP dilimleme ve INT8 nicelemesinden önce birleştirilir. İstek zamanı LoRA matris çarpımları yoktur. Kaynak formatları ve sabit programlar için [model hazırlama](../docs/weights.md) konusuna bakın.

<a id="serve"></a>

## Bir sunucu başlatın

Yalnızca başka bilgi işlem süreci olmayan cihazları kullanın. Bir kart için:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

Diğer arka uçlar aynı sunucu API'sini kullanır:

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

Seçilen cihazlarda aynı anda bir sunucu çalıştırın. Oluşturulan yerel vLLM-Omni komutunu bir model yüklemeden incelemek için `--dry-run` ekleyin.

Tek kartlı örnekler, INT8 DiT'yi yerleşik tutarken metin kodlayıcının akışını sağlar. Yerel tarif DiT'yi nicemler; metin kodlayıcı BF16 olarak kalır, dolayısıyla bellek ayak izinin yine de başlatma ilkesine dahil edilmesi gerekir. **BF16** için `--precision bf16`'yi kullanın; gerekirse `--offload layer --no-precompute-adaln`'yi açıkça seçin. INT8 yerel tarifi kullanır:

```json
{"transformer":{"method":"int8"}}
```

INT8 projeksiyonları BF16 Q/K/V'yi dikkati besler. Dikkat akümülatörler, Veda tahminleri ve VDN durumu, öngörülen hassasiyetlerini korur. Bu INT8 model çıkarımıdır, her dikkat işleminin INT8 olduğu iddiası değildir. Yerel tarif, BF16'de desteklenmeyen katmanları koruyor. Tek kartlı H3, 65.535'lik NPU nicemlenmiş matmul çıkış sınırını aşan 96.768 genişlikte bir projeksiyona sahiptir; TP, sıra başına genişliğini aralığa düşürebilir.

**dört kartlı çıkarım** için yerel tensör paralel kafa parçalamayı kullanın:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices`, 56 başlığı bölen 1, 2, 4, 7 ve 8'i destekler. Veda, parçalama sonrasında küresel tahminci kafa kimliklerini korur. Yedi kartlı DiT TP, 64 sorgu / 8 KV kafaları yediye bölünemediğinden tek metin kodlayıcı sıralaması kullanır. Varsayılan olarak `--offload none`'yi (yerleşik ağırlıklar) başlatır. Bellek yetersizse `--offload text`, DiT'yi cihazda tutarken yalnızca metin kodlayıcıyı aktarır; `--offload module` komple bileşenleri değiştirir (deneysel yeterlilik); `--offload layer`, biri aktif, diğeri önceden getirilmiş bloğu tutarak her iki bileşenin akışını sağlar. Aşağıdaki tamamlanmış eşleştirilmiş Boşaltma referansı, her iki donanım platformunda da açıkça `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) kullanır. Yeni uçtan uca ölçümler **yalnızca INT8** kullanır; yerleşik yürütme ile başlar ve gerekli her Boşaltma geri dönüşünü işaretler. Yalıtılmış dikkat ve DiT blok karşılaştırması, her iki cihazda da BF16'yi kullanır: 0/24/49 katmanları, her biri üç çalıştırma, dokuz zamanlamanın toplanması ve dokuza bölünmesi. İsteğe bağlı `--parallel ulysses`, VDN olmayan yollar için kullanıma sunulur; TP varsayılan olarak kalır. Yerel döşemeli VAE kod çözme artık seçilen tüm kartlarda varsayılan olarak kullanılıyor (`--vae-parallel-size 0`); tek bir kod çözücü sırası için `--vae-parallel-size 1` kullanın. INT8 Tam / 124 çerçeveli boru hattı bir, iki, dört ve sekiz VAE derecesiyle geçti. Diğer yöntem/kare sayısı kombinasyonları E2E matrisinde kalır.

Yerleşik NPU lansmanları, sunucu hazır olmadan önce AdaLN sonuçlarını hazırlar. Hazırlık, yüklenen modeli çözer ve LoRA'nin gerçek sigma konumlarını, DiT güncelleme sayısını ve video/ses geçişlerini birleştirir; sekiz adıma sabit kodlanmamıştır. Öğretmen/özel öğrenci başlatmayı belirtmek için `serve --steps N`'yi kullanın ve aynı sayıyı `generate --steps N`'de kullanın. Çakışan damıtılmış bir kontrol noktası programı reddedilir. Gerekli planlarla önbellek kapasitesi artar. Orijinal projeksiyon matematiği korunur; AdaLN ana ağırlıklar CPU'da kalırken, hazırlanan sonuçlar NPU'da kalır. TP4/TP8 aynı zamanda aşağı akış önbelleklerini geçersiz kılan düşük bitli FP32 azaltma varyasyonunu önlemek için tam zaman adımı girişlerinden yerel zaman yerleştirmelerini de belleğe alır. Ağırlıklar, sayısal ayarlar ve tüm sıralamalardaki isabet anlaşması hâlâ yeniden kullanımı koruyor. Farklı bir planlamayı önceden hesaplamak için yeni konfigürasyonla yeniden başlayın; hazırlıksız bir girdi orijinal hesaplamayı kullanır ve ağırlıkları aktarabilir. `--no-precompute-adaln` hazırlığı devre dışı bırakır; DiT modül/katman akışı, varsayılan olarak devre dışı kalmasını sağlar.

Tek kartlı 345 çerçeveli Spark/VSA/VDN için, `--offload layer --no-precompute-adaln`'yi orijinal doğrulanmış operatörlerle birlikte kullanın. E2E tablosu bu yerel katman Boşaltma hücrelerini †L ile işaretler; ‡R, önceki çalıştırmadan yeniden kullanılan eşleşen ölçümleri tanımlar. Diğer tüm hücreler, bireysel olarak kaydedilmiş ikamet politikalarını korur.

Başlatıcı, yerel çalışan isteğinin son tarihini 7200 saniyeye ayarlar, böylece uzun bir video, yukarı akış 600 saniyelik varsayılan değere göre durdurulmaz. Bu sınırı geçersiz kılmak için başlatmadan önce `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT`'yi ayarlayın; model hesaplamayı veya ölçülen istek süresini değil, bekleme davranışını değiştirir.

## İki 768p kasayı oluşturun

Bir UTF-8 bilgi istemi dosyası oluşturun, bir sunucu başlatın ve ardından eşleşen adım sayısını kullanın:

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

Sunucunun `--method`'sini istemciye iletin: beş yöntemin tümü (Full/VSA/Veda/VDN/Spark) varsayılan olarak sekiz gerçek iletimdir. İsteğe bağlı orijinal çok adımlı model için sunucuyla eşleşen `generate --method full --no-few-step-lora` (50 adım) kullanın. Ayrı olarak hazırlanmış bir topluluk öğrencisi için açık bir `--steps` mevcuttur. `--guidance-scale` isteğe bağlıdır; ihmal yerel politikayı korur. Hiçbir İyileştirme aşaması eklenmez. İstemci MP4'ü ve JSON zamanlama makbuzunu kaydeder.

<!-- BEGIN BF16 LAYER RESULTS -->

## BF16 katman karşılaştırması: her iki cihazda da yerleşik

Yalnızca 1344×768, 124/345 kare. Gerçek öğrenci ağırlıkları ve baytla aynı dondurulmuş sentetik gizli durumlar; bunlar bir video isteğinden elde edilen katman etkinleştirmeleri değildir. 0, 24 ve 49. katmanların her biri iki adet süresiz ısınma ve üç resmi koşu alır: **dokuz zamanlamayı toplayın ve dokuza bölün**. Ağırlıklar her iki cihazda da yerleşik kalır. Etkinleştirmeler/ana ağırlıklar BF16'dir; gerekli FP32 tahmin edicileri, birikimler ve VDN durumu korunur. Hızlandırmalar aynı donanıma sahip, aynı şekilli BF16 Tam temel çizgiyi kullanır. Göreceli olarak daha yüksek hızlanma kalın harflerle gösterilmiştir; mutlak gecikme de gösterilir.

### Yalnızca dikkat çekirdeği

Hazırlanan Q/K/V ile başlar ve dikkat çıkışında biter. QKV projeksiyonu, Q/K normu, RoPE, çıktı projeksiyonu, AdaLN ve FFN hariçtir. Gerekli seyrek kapıları, tahmini, seçimi, yeniden düzenlemeyi ve restorasyonu içerir; VDN hem tam softmax hem de doğrusal çekirdekleri içerir. Her yöntem bu sınırı kullanır; tüm modül zamanlamaları değiştirilmez. Çekirdek zamanlaması 10/10 eşleşen donanım çiftini tamamladı; bekleyen hücreler önceki modül ölçümlerini yeniden kullanmaz.

| Çerçeveler | Yöntem | Ascend gecikme | PRO 6000 gecikme | Ascend hızlandırma | PRO 6000 hızlandırma |
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

### DiT bloğunu tamamlayın

Projeksiyonlar, normalleştirme, RoPE, dikkat çekirdeği, AdaLN, FFN ve artık işlemler dahil olmak üzere blok girişinden blok çıkışına kadar çalışır. Çekirdekten bağımsız olarak ölçülmüştür; aralarındaki fark bir FFN ölçümü değildir ve ikisi de E2E süresine yansıtılmaz.

| Çerçeveler | Yöntem | Ascend gecikme | PRO 6000 gecikme | Ascend hızlandırma | PRO 6000 hızlandırma |
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

![BF16 dikkat çekirdeği ve DiT blok karşılaştırması](../docs/figures/bf16-three-layer.png)[Dokuz çalıştırmalı kayıtların tümü ve giriş kimlik kontrolleri](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1/2/4/8 kartlar

1344×768, sekiz gerçek DiT güncellemesi, metin kodlama, video/ses kod çözme ve MP4 yazma dahil. Yükleme, önceki LoRA birleştirme, AdaLN hazırlığı ve ısınmalar hariçtir. Yerleşik DiT'yi tercih edin; yerleşik satırlar, resmi talebin etrafındaki her aşamada sıfır AdaLN ıskalamayı ve ağırlık transferini doğrular. Kaydedilen OOM'lardan sonra, işaretlenmiş tek kartlı 345 çerçeveli hücreler, istendiği gibi yerel katman Aktarımı ve yeniden kullanım eşleşmesini, tamamen doğrulanmış önceki ölçümleri kullanır; bu hücreler yeni ölçümler olarak sunulmak yerine açıkça işaretlenir.

| Çerçeveler | Yöntem | 1 kart | 2 kart | 4 kart | 8 kart |
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

Kapalı **40/40**: 37 taze ve 3 yeniden kullanılmış önceki ölçüm. †T: metin kodlayıcı katmanı Yalnızca boşaltma; †L: yerel DiT/metin kodlayıcı katmanı Tek katmanlı önceden getirme ve yerleşik VAE ile boşaltma; ‡R: 2026-10-31'den alınan orijinal ölçümlerle eşleşiyor. İşaretlenmemiş hücrelerde çalışma zamanı Aktarımı yoktur. Yalnızca yerleşik satırlar AdaLN'yi önceden hesaplar; katman Boşaltma satırları, DiT yerleşimi veya sıfır AdaLN aktarımı iddiasında bulunmaz. VAE maç kartı sayısını sıralar; DiT kafaları 56/28/14/7'ye bölündü. Yerel INT8 kapsamı korunur.[INT8 E2E kayıtları](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

Daha önce eşleşen katman-Yük Boşaltma karşılaştırması[geçmişsel tek kart referansında](../docs/archive/matched-offload-single-card.md) korunur. Geçerli tablodaki yeniden kullanılan hücreler, değiştirilmemiş orijinal kayıtlara bağlantı verir.

## Niteliksel sonuçlar ve doğrulama

Yalnızca Ascend 910B3: **1344×768, 345 yerel kare, 24 FPS (14,375 saniye, yaklaşık 14,4 saniye)**, sesli.

Beş yöntemin tümü sekiz ileriyi kullanır. Sol: BF16. Sağ: yerel W8A8 INT8. Tüm örnekler tek bir kart, aynı komut istemi, ortak tohum **7** ve eşleşen Boşaltma kullanır.

> Bir bisikletçi sessiz, yeşil bir park yolunda ilerliyor, pürüzsüz sinematik izleme çekimi, doğal gün ışığı.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Tam + LightX2V8 · 8 adım**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Tam + LightX2V8 · 8 adım**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · %80 · 8 adım**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · %80 · 8 adım**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · %90 · 8 adım**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · %90 · 8 adım**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · resmi · 8 adım**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · resmi · 8 adım**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 adım**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 adım**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2, %80 video seyrekliği kullanır; Veda %90 kullanır. Metin ve ses kendi yerel kurallarını korur.[On videonun tümü için parametreler, SHA256 ve kaynak](../examples/videos/qualitative-1344-345.json). Tam kod çözme, 345 kareyi ve sesi doğrular; Klip başına 16 tekdüze kare görsel olarak incelendi. Farklı öğrencilerin piksel veya algısal olarak eşdeğer olduğu iddia edilmez. VSA, atış değişikliklerini korur; titreşimsizlik garantisi verilmemektedir.

İlk entegrasyon, referans videoları olmayan toplu bir T2VA'yı hedefliyor. Bkz.[mimari](../docs/architecture.md),[doğrulama durumu](../docs/validation.md) ve [üçüncü taraf bildirimleri](../THIRD_PARTY_NOTICES.md).
