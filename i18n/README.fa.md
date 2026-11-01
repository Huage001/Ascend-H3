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

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | **فارسی** | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

<div dir="rtl">

> این README کامل یک ترجمه به کمک هوش مصنوعی از README انگلیسی در commit be5c42e03f9a است. ممکن است منسوخ شود. برای آخرین اطلاعات به نسخه انگلیسی مراجعه کنید. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

استنتاج MiniMax H3 در Ascend با توجه کامل، VSA، Veda، VDN و Spark-H3 در یک مخزن. این ادغام از بارگذاری مدل vLLM-Omni، کوانتیزاسیون ترانسفورماتور INT8 و سرهای توجه موازی تانسور بومی استفاده می کند.

**حوزه تایید شده:** سلول های 40/40 INT8 E2E در 768×1344 فریم، 124/345 فریم و 1/2/4/8 NPU (37 اندازه گیری جدید و 3 سلول لایه تاریخی مشخص شده-Offload). مقایسه مقیم BF16 در Ascend 910B3 و PRO 6000 شامل 20 اندازه گیری هسته توجه و 20 اندازه گیری بلوک کامل است. اپراتورهای توجه نسخه های معتبر خود را حفظ می کنند.

## بارهای کاری پیش فرض 768p

<div dir="ltr">

| مورد | عرض × ارتفاع | فریم های خروجی بومی | شبکه ویدیویی پنهان |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

</div>

FPS خروجی 24 است. درخواست‌ها از تعداد فریم صریح استفاده می‌کنند و درون یابی فریم غیرفعال است. مورد طولانی، حجم کاری 345 فریم بومی توافق شده است. به 341 فریم برش داده نمی شود. BF16 و INT8 از این دو بار کاری 1344×768 استفاده می کنند. حذف `--case` `1344-124` را انتخاب می کند.

<a id="install"></a>

## نصب و راه اندازی

از Linux، Python 3.12 و محیط سازگار Ascend Torch/CANN استفاده کنید. پشته vLLM، vLLM-Ascend و vLLM-Omni پین شده را که در[یادداشت های محیطی](../docs/environment.md) توضیح داده شده است، نصب کنید. نسخه های فریمورک نامرتبط را با هم ترکیب نکنید. سپس این مخزن را از منبع نصب کنید:

<div dir="ltr">

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

</div>

بیلدها فاقد وزن مدل هستند و بدون اشغال NPU اجرا می شوند. کتابخانه های بومی به صورت محلی گردآوری شده و خارج از صندوق نگهداری می شوند. برای یک باطن، `all` را با `vsa`، `veda` یا `spark` جایگزین کنید. Full و VDN به این پسوندهای سفارشی CANN نیاز ندارند. تسویه حساب قابل ویرایش را در دسترس نگه دارید.

<a id="models"></a>

## نقاط بازرسی و انتخاب الگوریتم

<div dir="ltr">

| روش | دارایی های مدل مورد نیاز | محاسبات |
| --- | --- | --- |
| `full` | پایه H3; LightX2V 8 مرحله ای به طور پیش فرض، یا یک انجمن LoRA | توجه متراکم |
| `vsa` | دانشجوی رسمی FastH3 V2، شامل تمام 50 گیت فشرده سازی آموزش دیده | 8 فوروارد، 80% پراکندگی ویدئو؛ KV64 منطقی، بلوک‌های KV انتخاب شده با KV128 روی تراشه جفت شدند |
| `veda` | پایه H3 + LightX2V 8 مرحله ای + پیش بینی رسمی آموزش دیده Veda | مسیرهای بلوک پویا در هر سر Veda، پراکندگی اسمی 90٪ |
| `vdn` | پایه H3 + دایرکتوری رسمی پست بازرسی مرحله-DMD VDN | سافت مکس محلی به همراه 128 بعدی شاخه دلتا نت آموخته شده |
| `spark` | پایه H3 + LightX2V 8 پله | پارتیشن سلسله مراتبی Spark-H3 و softmax دقیق/پراکسی مشترک، نامی 90% پراکندگی شاخه دقیق |

</div>

مدل ها را تحت مجوز اصلی خود دانلود کنید. وزن ها دوباره توزیع نمی شوند. دانش آموزان مقطر مختلف برنامه ها و قوانین آموخته شده متفاوتی دارند. تغییر توجه روی وزن های دلخواه تضمین کننده کیفیت تطابق نیست. VSA به گیت های آموزش دیده غیر صفر نیاز دارد. VDN به وزن شاخه های آموخته شده خود نیاز دارد. دستورات دانلود و آماده سازی در[مدل دارایی](../docs/weights.md) هستند.

Full، Veda و Spark به طور پیش فرض **LightX2V هشت مرحله ای LoRA** پین شده را فعال می کنند. از `--no-few-step-lora` برای اجرای مدل اصلی چند مرحله ای استفاده کنید. یک جامعه آماده `--lora /models/style` قبل از اتمام بارگیری با LightX2V ترکیب می شود. هنگام انتخاب صریح دانش آموز انجمن دیگری، دانش آموز پیش فرض را غیرفعال کنید. Veda همیشه به پیش بینی رسمی آموزش دیده خود نیاز دارد. VSA فقط پست بازرسی کامل رسمی **FastH3 V2** را می پذیرد: هشت جلو، 80% پراکندگی ویدئو، تغییرات ویدئو/صوت 10/3، tile64، راهنمایی 1. وزن های منتشر شده را مستقیماً بارگذاری می کند و `--lora` را نمی گیرد. VDN فقط ایست بازرسی کامل رسمی مرحله-DMD خود را می پذیرد. هیچ یک انجمن اضافی LoRA یا پرچم غیرفعال کردن دانش آموز را نمی پذیرد.

<div dir="ltr">

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

</div>

منبع رسمی PEFT یک بار با استفاده از آلفا/رتبه اعلام شده خود عادی می شود. فایل های آماده شده هش چک می شوند. همه آداپتورهای انتخاب شده قبل از برش TP و کمی سازی INT8 ادغام می شوند. ضرب ماتریس LoRA در زمان درخواست وجود ندارد. برای قالب‌های منبع و زمان‌بندی‌های ثابت به[آماده‌سازی مدل](../docs/weights.md) مراجعه کنید.

<a id="serve"></a>

## یک سرور راه اندازی کنید

فقط از دستگاه هایی استفاده کنید که هیچ فرآیند محاسباتی دیگری ندارند. برای یک کارت:

<div dir="ltr">

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

</div>

سایر پشتیبان ها از همان API سرور استفاده می کنند:

<div dir="ltr">

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

</div>

یک سرور را در یک زمان در دستگاه های انتخاب شده اجرا کنید. `--dry-run` را اضافه کنید تا دستور بومی تولید شده vLLM-Omni را بدون بارگیری مدل بررسی کنید.

نمونه‌های تک کارتی رمزگذار متن را جریان می‌دهند در حالی که INT8 DiT ساکن هستند. دستور پخت بومی DiT را کوانتیزه می کند. رمزگذار متن BF16 باقی می ماند، بنابراین ردپای حافظه آن همچنان باید در خط مشی راه اندازی گنجانده شود. برای **BF16**، از `--precision bf16` استفاده کنید. در صورت لزوم `--offload layer --no-precompute-adaln` را به صراحت انتخاب کنید. INT8 از دستور پخت بومی استفاده می کند:

<div dir="ltr">

```json
{"transformer":{"method":"int8"}}
```

</div>

پیش بینی های INT8 BF16 Q/K/V را مورد توجه قرار می دهد. انباشته های توجه، پیش بینی های Veda و وضعیت VDN دقت تجویز شده خود را حفظ می کنند. این استنتاج مدل INT8 است، نه ادعایی که هر عملیات توجهی INT8 است. دستور العمل اصلی لایه های پشتیبانی نشده را در BF16 حفظ می کند. تک کارت H3 دارای یک پروجکشن با عرض 96768 است که از حد خروجی matmul کوانتیزه NPU 65535 فراتر می رود. TP می تواند عرض هر رتبه خود را به محدوده کاهش دهد.

برای **استنتاج چهار کارتی**، از تقسیم بندی سر موازی تانسور بومی استفاده کنید:

<div dir="ltr">

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

</div>

`--devices` از 1، 2، 4، 7 و 8 پشتیبانی می کند که 56 هد را تقسیم می کنند. Veda شناسه های هد پیش بینی کننده جهانی را پس از به اشتراک گذاری حفظ می کند. هفت کارتی DiT TP از یک رتبه رمزگذار متن استفاده می کند زیرا 64 پرس و هد 8 کیلو ولت آن را نمی توان بر هفت تقسیم کرد. به طور پیش فرض روی `--offload none` (وزن ساکن) راه اندازی می شود. اگر حافظه کافی نباشد، `--offload text` تنها رمزگذار متن را استریم می کند در حالی که DiT را روی دستگاه نگه می دارد. `--offload module` اجزای کامل را مبادله می کند (شرایط آزمایشی). `--offload layer` هر دو مؤلفه را استریم می کند و یکی را فعال و دیگری را از پیش واکشی می کند. مرجع تطبیق-Offload تکمیل شده زیر به صراحت از `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) در هر دو پلت فرم سخت افزاری استفاده می کند. اندازه‌گیری‌های انتها به انتها از **فقط INT8** استفاده می‌کنند که با اجرای مقیم شروع می‌شود و هر Offload بازگشتی لازم را علامت‌گذاری می‌کند. توجه ایزوله و مقایسه بلوک DiT از BF16 در هر دو دستگاه استفاده می‌کند: لایه‌های 0/24/49، هر کدام سه اجرا، جمع‌بندی نه زمان‌بندی و تقسیم بر 9. `--parallel ulysses` اختیاری برای مسیرهای غیر VDN قرار می گیرد. TP پیش فرض باقی می ماند. رمزگشایی VAE کاشی‌شده بومی اکنون برای همه کارت‌های انتخابی پیش‌فرض است (`--vae-parallel-size 0`). از `--vae-parallel-size 1` برای یک رتبه رمزگشا استفاده کنید. خط لوله INT8 کامل / 124 فریم با رتبه های یک، دو، چهار و هشت VAE به پایان رسیده است. سایر ترکیبات روش / تعداد فریم در ماتریس E2E باقی می مانند.

راه اندازی های Resident NPU نتایج AdaLN را قبل از آماده شدن سرور آماده می کند. آماده سازی مدل بارگذاری شده را حل می کند و موقعیت های سیگما واقعی LoRA، تعداد به روز رسانی DiT و تغییرات ویدیو/صوتی را ادغام می کند. تا هشت مرحله سخت کدگذاری نشده است. از `serve --steps N` برای تعیین راه اندازی معلم/دانش آموز سفارشی استفاده کنید و از همان تعداد در `generate --steps N` استفاده کنید. برنامه ایست بازرسی مقطر متناقض رد می شود. ظرفیت کش با برنامه های مورد نیاز افزایش می یابد. ریاضی طرح ریزی اصلی حفظ می شود. وزن های اصلی AdaLN روی CPU باقی می مانند، در حالی که نتایج آماده شده روی NPU باقی می مانند. TP4/TP8 همچنین جاسازی‌های زمانی بومی را از ورودی‌های مرحله زمانی دقیق به خاطر می‌سپارد تا از تغییرات کاهشی FP32 با بیت پایین که کش‌های پایین‌دستی را باطل می‌کند، جلوگیری کند. وزن‌ها، تنظیمات عددی و توافقنامه تمام رتبه‌ای همچنان از استفاده مجدد محافظت می‌کنند. با پیکربندی جدید مجدداً راه اندازی کنید تا برنامه زمانی متفاوتی را از قبل محاسبه کنید. یک ورودی آماده نشده از محاسبات اصلی استفاده می کند و ممکن است وزن ها را منتقل کند. `--no-precompute-adaln` آماده سازی را غیرفعال می کند. پخش ماژول/لایه DiT آن را به طور پیش فرض غیرفعال نگه می دارد.

برای Spark/VSA/VDN تک کارتی 345 فریم، از `--offload layer --no-precompute-adaln` با اپراتورهای معتبر اصلی استفاده کنید. جدول E2E این سلول های Offload لایه بومی را با †L علامت گذاری می کند. ‡R اندازه‌گیری‌های تطبیقی ​​را که از اجرای قبلی مجدداً استفاده شده‌اند، شناسایی می‌کند. همه سلول های دیگر خط مشی اقامت ثبت شده خود را حفظ می کنند.

راه‌انداز مهلت درخواست کارگر بومی را روی 7200 ثانیه تنظیم می‌کند تا یک ویدیوی طولانی با پیش‌فرض بالادستی 600 ثانیه‌ای لغو نشود. `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` را قبل از راه‌اندازی تنظیم کنید تا این محدودیت را لغو کنید. رفتار انتظار را تغییر می دهد، نه محاسبات مدل یا مدت زمان اندازه گیری درخواست را.

## دو مورد 768p را ایجاد کنید

یک فایل اعلان UTF-8 ایجاد کنید، یک سرور راه‌اندازی کنید، سپس از تعداد مراحل منطبق آن استفاده کنید:

<div dir="ltr">

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

</div>

`--method` سرور را به مشتری ارسال کنید: هر پنج روش (Full/VSA/Veda/VDN/Spark) به طور پیش‌فرض به هشت فوروارد واقعی می‌رسد. برای مدل چند مرحله‌ای اصلی اختیاری، از `generate --method full --no-few-step-lora` (50 مرحله)، مطابق با سرور استفاده کنید. یک `--steps` صریح برای یک دانش‌آموز جامعه که جداگانه آماده شده است، در دسترس است. `--guidance-scale` اختیاری است. حذف سیاست بومی را حفظ می کند. مرحله Refine اضافه نشده است. مشتری MP4 و رسید زمان بندی JSON را ذخیره می کند.

<!-- BEGIN BF16 LAYER RESULTS -->

## مقایسه لایه BF16: ساکن در هر دو دستگاه

1344×768، فقط 124/345 فریم. وزن‌های واقعی دانش‌آموز و حالت‌های مخفی مصنوعی منجمد مشابه بایت. اینها فعال‌سازی لایه‌ای نیستند که از یک درخواست ویدیویی گرفته شده‌اند. لایه های 0، 24 و 49 هر کدام دو گرم کردن بدون زمان و سه اجرای رسمی دریافت می کنند: ** نه زمان بندی را جمع کنید و بر 9 تقسیم کنید **. وزن ها در هر دو دستگاه ثابت می مانند. فعال‌سازی‌ها/وزن‌های اصلی BF16 هستند. پیش‌بینی‌کننده‌های مورد نیاز FP32، انباشتگی‌ها و وضعیت VDN حفظ می‌شوند. Speedup ها از خط پایه کامل BF16 با سخت افزار یکسان استفاده می کنند. سرعت نسبی بالاتر پررنگ است. تأخیر مطلق نیز نشان داده شده است.

### فقط هسته توجه

با Q/K/V آماده شروع می شود و به خروجی توجه ختم می شود. شامل طرح ریزی QKV، هنجار Q/K، RoPE، طرح ریزی خروجی، AdaLN و FFN نمی شود. شامل دروازه‌های پراکنده، پیش‌بینی، انتخاب، بازآرایی و بازسازی است. VDN شامل هر دو هسته softmax کامل و خطی است. هر روشی از این مرز استفاده می کند. زمان بندی کل ماژول جایگزین نمی شود. زمان بندی هسته 10/10 جفت سخت افزار منطبق را تکمیل کرده است. سلول های معلق از اندازه گیری های قبلی ماژول استفاده مجدد نمی کنند.

<div dir="ltr">

| قاب ها | روش | تاخیر Ascend | تأخیر PRO 6000 | افزایش سرعت Ascend | افزایش سرعت PRO 6000 |
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

</div>

### بلوک DiT را کامل کنید

از ورودی بلوک به بلوک خروجی، از جمله پیش بینی، نرمال سازی، RoPE، هسته توجه، AdaLN، FFN و عملیات باقیمانده اجرا می شود. مستقل از هسته اندازه گیری می شود. تفاوت آنها یک اندازه گیری FFN نیست و به زمان E2E نیز تعمیم داده نمی شود.

<div dir="ltr">

| قاب ها | روش | تاخیر Ascend | تأخیر PRO 6000 | افزایش سرعت Ascend | افزایش سرعت PRO 6000 |
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

</div>

![BF16 هسته توجه و مقایسه بلوک DiT](../docs/figures/bf16-three-layer.png)[همه سوابق نه اجرا شده و بررسی هویت ورودی](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1 / 2 / 4 / 8 کارت

768×1344، هشت به‌روزرسانی واقعی DiT، از جمله رمزگذاری متن، رمزگشایی ویدیو/صوت و نوشتن MP4. در حال بارگذاری، ادغام قبلی LoRA، آماده سازی AdaLN و گرم کردن حذف شده است. ترجیحاً ساکن DiT؛ ردیف های ساکن صفر AdaLN و انتقال وزن را در هر رتبه در مورد درخواست رسمی تأیید می کنند. پس از OOM های ثبت شده، سلول های 345 فریم تک کارتی علامت گذاری شده از لایه بومی Offload و استفاده مجدد از تطبیق استفاده می کنند، اندازه گیری های قبلی کاملاً تأیید شده طبق درخواست. این سلول ها به جای اینکه به عنوان اندازه گیری های تازه ارائه شوند، به صراحت علامت گذاری می شوند.

<div dir="ltr">

| قاب ها | روش | 1 کارت | 2 کارت | 4 کارت | 8 کارت |
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

</div>

پوشیده شده **40/40**: 37 اندازه گیری تازه و 3 مورد استفاده مجدد. †T: لایه متن-رمزگذار فقط Offload; †L: لایه بومی DiT/تکست-انکودر Offload با prefetch یک لایه و ساکن VAE. ‡R: مطابق با اندازه‌گیری‌های اصلی از 2026-10-31. سلول های بدون علامت زمان اجرا Offload ندارند. فقط ردیف های ساکن AdaLN را از پیش محاسبه می کنند. ردیف‌های تخلیه لایه هیچ ادعایی برای اقامت DiT یا صفر AdaLN برای انتقال ندارند. VAE رتبه های منطبق با تعداد کارت. تقسیم سرهای DiT 56/28/14/7. پوشش بومی INT8 حفظ می شود.[سوابق INT8 E2E](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

مقایسه قبلی لایه-Offload در[مرجع تک کارتی تاریخی](../docs/archive/matched-offload-single-card.md) حفظ شده است. سلول‌های استفاده مجدد در جدول فعلی به رکوردهای اصلی بدون تغییر پیوند دارند.

## نتایج کیفی و اعتبار سنجی

فقط Ascend 910B3: **1344×768، 345 فریم بومی، 24 FPS (14375 ثانیه، تقریباً 14.4 ثانیه)**، همراه با صدا.

هر پنج روش از هشت فوروارد استفاده می کنند. سمت چپ: BF16. سمت راست: بومی W8A8 INT8. همه نمونه ها از یک کارت، همان اعلان، دانه مشترک **7** و Offload همسان استفاده می کنند.

> دوچرخه سواری در امتداد مسیر پارک سبز و آرام، شات ردیابی سینمایی صاف، نور طبیعی روز حرکت می کند.

<!-- BEGIN QUALITATIVE GALLERY -->
<table dir="ltr">
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Full + LightX2V8 · 8 مرحله**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Full + LightX2V8 · 8 مرحله**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 مرحله**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 مرحله**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 مرحله**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 مرحله**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · رسمی · 8 مرحله**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · رسمی · 8 مرحله**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 مرحله**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 مرحله**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 از 80% پراکندگی ویدئو استفاده می کند. Veda از 90٪ استفاده می کند. متن و صدا قوانین اصلی خود را حفظ می کنند.[پارامترها، SHA256 و منشأ هر ده ویدیو](../examples/videos/qualitative-1344-345.json). رمزگشایی کامل 345 فریم و صدا را تأیید می کند. 16 فریم یکنواخت در هر کلیپ به صورت بصری بررسی شد. دانش‌آموزان متمایز از نظر پیکسلی یا ادراکی معادل نیستند. VSA تغییرات شات را حفظ می کند. هیچ تضمینی بدون سوسو زدن وجود ندارد.

ادغام اولیه دسته‌ای T2VA بدون ویدیوهای مرجع را هدف قرار می‌دهد.[معماری](../docs/architecture.md)،[وضعیت اعتبارسنجی](../docs/validation.md) و[اعلامیه های شخص ثالث](../THIRD_PARTY_NOTICES.md) را ببینید.

</div>
