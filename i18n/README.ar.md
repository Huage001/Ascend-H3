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

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | **العربية** | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> **License:** Project-authored contributions in this distribution are offered under the [PolyForm Noncommercial License 1.0.0](../LICENSE). Third-party code and model weights have separate terms, and earlier Apache-2.0 grants remain in effect. See [third-party notices](../THIRD_PARTY_NOTICES.md). For uses not covered by the applicable terms, contact [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn).

<div dir="rtl">

> هذا الملف التمهيدي الكامل عبارة عن ترجمة بمساعدة الذكاء الاصطناعي للملف التمهيدي باللغة الإنجليزية الموجود في الالتزام be5c42e03f9a. قد تصبح قديمة. راجع النسخة الإنجليزية للحصول على أحدث المعلومات. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

استنتاج MiniMax H3 على Ascend مع الاهتمام الكامل وVSA وVeda وVDN وSpark-H3 في مستودع واحد. يعيد التكامل استخدام تحميل نموذج vLLM-Omni، وتكميم المحولات INT8 ورؤوس الاهتمام المتوازية الأصلية.

**النطاق الذي تم التحقق منه:** 40/40 خلايا INT8 E2E بدقة 1344×768 و124/345 إطارًا و1/2/4/8 وحدات NPU (37 قياسًا جديدًا و3 خلايا إلغاء تحميل الطبقة التاريخية المميزة بوضوح). تحتوي مقارنة BF16 المقيمة على Ascend 910B3 وPRO 6000 على 20 قياسًا أساسيًا للانتباه و20 قياسًا كاملاً. يحتفظ مشغلو الاهتمام بإصداراتهم التي تم التحقق من صحتها.

## أعباء العمل الافتراضية 768p

<div dir="ltr">

| القضية | العرض × الارتفاع | إطارات الإخراج الأصلية | شبكة الفيديو الكامنة |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

</div>

إخراج FPS هو 24. تستخدم الطلبات عددًا صريحًا من الإطارات، مع تعطيل استيفاء الإطارات. الحالة الطويلة هي حمل العمل الأصلي المتفق عليه والذي يبلغ 345 إطارًا؛ لم يتم اقتصاصه إلى 341 إطارًا. يستخدم BF16 وINT8 أحمال العمل 1344×768 هذين. يؤدي حذف `--case` إلى تحديد `1344-124`.

<a id="install"></a>

## التثبيت

استخدم Linux وPython 3.12 وبيئة Ascend Torch/CANN المتوافقة. قم بتثبيت مكدس vLLM وvLLM-Ascend وvLLM-Omni الموضح في[ملاحظات البيئة](../docs/environment.md). لا تجمع بين إصدارات إطار العمل غير ذات الصلة. ثم قم بتثبيت هذا المستودع من المصدر:

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

لا تحتوي الإصدارات على أوزان نموذجية وتعمل دون شغل وحدة NPU. يتم تجميع المكتبات الأصلية محليًا ويتم الاحتفاظ بها خارج منطقة الدفع. للحصول على واجهة خلفية واحدة، استبدل `all` بـ `vsa` أو `veda` أو `spark`؛ لا يتطلب Full وVDN امتدادات CANN المخصصة هذه. احتفظ بالخروج القابل للتحرير متاحًا.

<a id="models"></a>

## نقاط التفتيش واختيارات الخوارزمية

<div dir="ltr">

| الطريقة | الأصول النموذجية المطلوبة | الحساب |
| --- | --- | --- |
| `full` | قاعدة H3؛ LightX2V 8 خطوات بشكل افتراضي، أو مجتمع LoRA | اهتمام كثيف |
| `vsa` | استكمال طالب FastH3 V2 الرسمي، بما في ذلك جميع بوابات الضغط المدربة البالغ عددها 50 | 8 مهاجمين، 80% من تناثر الفيديو؛ منطقي KV64، كتل KV محددة مقترنة بـ KV128 على الشريحة |
| `veda` | قاعدة H3 + LightX2V 8 خطوات + جهاز توقع Veda الرسمي المُدرب | مسارات الكتلة الديناميكية Veda لكل رأس، متناثرة اسمية بنسبة 90% |
| `vdn` | قاعدة H3 + دليل نقاط تفتيش VDN الرسمي | softmax المحلي بالإضافة إلى فرع DeltaNet المتعلم ذو 128 بُعدًا |
| `spark` | قاعدة H3 + LightX2V 8 خطوات | Spark-H3 قسم هرمي ومفصل softmax دقيق/وكيل، اسمي 90% متناثر للفرع الدقيق |

</div>

قم بتنزيل النماذج بموجب تراخيصها الأصلية. لا يتم إعادة توزيع الأوزان. الطلاب المقطرون المختلفون لديهم جداول زمنية مختلفة وقواعد متعلمة؛ إن تبديل الاهتمام بالأوزان العشوائية لا يضمن جودة المطابقة. يتطلب VSA بوابات مدربة غير صفرية. يتطلب VDN أوزان الفروع التي تم تعلمها. أوامر التنزيل والتحضير موجودة في[أصول النموذج](../docs/weights.md).

يعمل كل من Veda وSpark على تمكين **LightX2V LoRA المكون من ثماني خطوات بشكل افتراضي. استخدم `--no-few-step-lora` لتشغيل النموذج الأصلي متعدد الخطوات. يتم دمج المجتمع المجهز `--lora /models/style` مع LightX2V قبل انتهاء التحميل؛ قم بتعطيل الطالب الافتراضي عند تحديد طالب مجتمع آخر بشكل صريح. يحتاج Veda دائمًا إلى المتنبئ الرسمي المدرب. يقبل VSA فقط النقطة الرسمية الكاملة **FastH3 V2 الكاملة**: ثمانية مهاجمين، 80% فيديو متناثر، إزاحات الفيديو/الصوت 10/3، Tile64، التوجيه 1. يقوم بتحميل الأوزان التي تم إصدارها مباشرة ولا يأخذ `--lora`. يقبل VDN فقط نقطة تفتيش DMD الرسمية الكاملة الخاصة به. ولا يقبل أي منهما مجتمعًا إضافيًا LoRA أو علامة تعطيل الطالب.

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

تتم تسوية مصدر PEFT الرسمي بمجرد استخدام ألفا/رتبة ألفا المعلنة؛ يتم فحص الملفات المعدة. يتم دمج كافة المحولات المحددة قبل تقطيع TP وتكميم INT8. لا توجد مضاعفات مصفوفة LoRA في وقت الطلب. راجع[إعداد النموذج](../docs/weights.md) للتعرف على تنسيقات المصدر والجداول الزمنية الثابتة.

<a id="serve"></a>

## بدء تشغيل الخادم

استخدم فقط الأجهزة التي لا تحتوي على عمليات حسابية أخرى. لبطاقة واحدة:

<div dir="ltr">

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

</div>

تستخدم الواجهات الخلفية الأخرى نفس واجهة برمجة تطبيقات الخادم:

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

قم بتشغيل خادم واحد في كل مرة على الأجهزة المحددة. أضف `--dry-run` لفحص أمر vLLM-Omni الأصلي الذي تم إنشاؤه دون تحميل نموذج.

تقوم الأمثلة ذات البطاقة الواحدة ببث برنامج تشفير النص مع الاحتفاظ بالمقيم INT8 DiT. الوصفة الأصلية تقيس DiT؛ يظل برنامج تشفير النص BF16، لذلك لا يزال يتعين تضمين بصمة الذاكرة الخاصة به في سياسة الإطلاق. بالنسبة إلى **BF16**، استخدم `--precision bf16`؛ إذا لزم الأمر، حدد `--offload layer --no-precompute-adaln` بشكل صريح. يستخدم INT8 الوصفة الأصلية:

<div dir="ltr">

```json
{"transformer":{"method":"int8"}}
```

</div>

تعمل إسقاطات INT8 على تغذية BF16 Q/K/V بالانتباه. تحافظ مراكم الانتباه وتنبؤات Veda وحالة VDN على دقتها الموصوفة. هذا هو استنتاج نموذج INT8، وليس ادعاء بأن كل عملية انتباه هي INT8. تحتفظ الوصفة الأصلية بطبقات غير مدعومة في BF16. تحتوي البطاقة الواحدة H3 على عرض يبلغ 96,768 عرضًا يتجاوز حد إخراج matmul الكمي لوحدة NPU وهو 65,535؛ يمكن أن يقلل TP عرضه لكل رتبة إلى النطاق.

بالنسبة إلى **الاستدلال المكون من أربع بطاقات**، استخدم تقسيم الرأس المتوازي الأصلي:

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

`--devices` يدعم 1، 2، 4، 7 و 8، والتي تقسم 56 رأسًا. يحتفظ Veda بمعرفات رأس التوقع العالمية بعد التقسيم. يستخدم DiT TP ذو السبع بطاقات رتبة واحدة لتشفير النص لأنه لا يمكن تقسيم رؤوس الاستعلام 64 / 8 كيلو فولت على سبعة. يتم تشغيله افتراضيًا على `--offload none` (الأوزان المقيمة). إذا كانت الذاكرة غير كافية، يقوم `--offload text` ببث برنامج تشفير النص فقط مع الاحتفاظ بـ DiT على الجهاز؛ يقوم `--offload module` بمبادلة المكونات الكاملة (التأهيل التجريبي)؛ يقوم `--offload layer` ببث كلا المكونين، مع الاحتفاظ بكتلة واحدة نشطة وأخرى تم جلبها مسبقًا. يستخدم مرجع إلغاء التحميل المطابق المكتمل أدناه بوضوح `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) على كلا النظامين الأساسيين للأجهزة. تستخدم القياسات الشاملة الجديدة **INT8 فقط**، بدءًا من التنفيذ المضمن ووضع علامة على كل إجراء احتياطي ضروري لإلغاء التحميل. يستخدم الاهتمام المعزول ومقارنة كتلة DiT BF16 على كلا الجهازين: الطبقات 0/24/49، ثلاثة أشواط لكل منها، وجمع التوقيتات التسعة والقسمة على تسعة. يتم عرض `--parallel ulysses` الاختياري للمسارات غير VDN؛ يبقى TP هو الافتراضي. أصبح فك تشفير VAE الأصلي المتجانب الآن افتراضيًا لجميع البطاقات المحددة (`--vae-parallel-size 0`)؛ استخدم `--vae-parallel-size 1` لرتبة وحدة فك ترميز واحدة. لقد مر خط الأنابيب INT8 الكامل / 124 إطارًا برتب واحدة واثنتين وأربعة وثمانية VAE. تبقى مجموعات الطرق/عدد الإطارات الأخرى في مصفوفة E2E.

تطلق وحدة NPU المقيمة إعداد نتائج AdaLN قبل جاهزية الخادم. يعمل التحضير على حل النموذج الذي تم تحميله ومواضع سيجما الفعلية لـ LoRA المدمجة وعدد تحديثات DiT وإزاحات الفيديو/الصوت؛ لم يتم ترميزه إلى ثماني خطوات. استخدم `serve --steps N` لتحديد تشغيل المعلم/الطالب المخصص، واستخدم نفس العدد في `generate --steps N`. تم رفض جدول نقاط التفتيش المقطر المتعارض. سعة ذاكرة التخزين المؤقت تنمو مع الخطط المطلوبة. يتم الاحتفاظ بحسابات الإسقاط الأصلية؛ تظل الأوزان الرئيسية AdaLN على وحدة المعالجة المركزية (CPU)، بينما تظل النتائج المعدة على وحدة NPU. يقوم TP4/TP8 أيضًا بحفظ تضمينات الوقت الأصلي من مدخلات الخطوات الزمنية الدقيقة لتجنب تباين تقليل FP32 منخفض البت الذي يؤدي إلى إبطال ذاكرة التخزين المؤقت النهائية. لا تزال الأوزان والإعدادات الرقمية واتفاق الضربات الشاملة تحمي إعادة الاستخدام. أعد التشغيل باستخدام التكوين الجديد لحساب جدول مختلف مسبقًا؛ يستخدم الإدخال غير المجهز الحساب الأصلي وقد ينقل الأوزان. `--no-precompute-adaln` يعطل الإعداد؛ يبقي تدفق الوحدة/الطبقة DiT معطلاً بشكل افتراضي.

بالنسبة إلى Spark/VSA/VDN أحادية البطاقة ذات 345 إطارًا، استخدم `--offload layer --no-precompute-adaln` مع عوامل التشغيل الأصلية التي تم التحقق من صحتها. يحدد الجدول E2E خلايا إلغاء تحميل الطبقة الأصلية هذه بـ †L. تحتفظ جميع الخلايا الأخرى بسياسة الإقامة المسجلة بشكل فردي.

يقوم المشغل بتعيين الموعد النهائي لطلب العامل الأصلي على 7200 ثانية حتى لا يتم إحباط مقطع فيديو طويل بواسطة الإعداد الافتراضي البالغ 600 ثانية. قم بتعيين `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` قبل الإطلاق لتجاوز هذا الحد؛ فهو يغير سلوك الانتظار، وليس حساب النموذج أو مدة الطلب المقاسة.

## قم بإنشاء حالتين 768p

أنشئ ملف مطالبة UTF-8، وابدأ الخادم، ثم استخدم عدد الخطوات المطابق له:

<div dir="ltr">

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

</div>

قم بتمرير `--method` الخاص بالخادم إلى العميل: جميع الطرق الخمس (Full/VSA/Veda/VDN/Spark) افتراضيًا إلى ثمانية عمليات إعادة توجيه فعلية. بالنسبة للنموذج الأصلي الاختياري متعدد الخطوات، استخدم `generate --method full --no-few-step-lora` (50 خطوة)، لمطابقة الخادم. يبقى `--steps` صريحًا متاحًا لطالب مجتمع مُجهز بشكل منفصل. `--guidance-scale` اختياري؛ الإغفال يحافظ على السياسة المحلية. لا تتم إضافة أي مرحلة صقل. يقوم العميل بحفظ ملف MP4 وإيصال توقيت JSON.

<!-- BEGIN BF16 LAYER RESULTS -->

## مقارنة طبقة BF16: موجودة على كلا الجهازين

1344×768، 124/345 إطارًا فقط. أوزان الطلاب الحقيقية والحالات المخفية الاصطناعية المجمدة المتطابقة بالبايت؛ هذه ليست عمليات تنشيط للطبقة تم التقاطها من طلب فيديو. تتلقى كل من الطبقات 0 و24 و49 إحماءين غير محددين وثلاثة أشواط رسمية: **اجمع التوقيتات التسعة واقسمها على تسعة**. تظل الأوزان موجودة على كلا الجهازين. التنشيطات/الأوزان الرئيسية هي BF16؛ يتم الاحتفاظ بتنبؤات FP32 المطلوبة والتراكمات وحالة VDN. تستخدم عمليات التسريع نفس الأجهزة ونفس الشكل BF16 خط الأساس الكامل. يعتبر التسارع النسبي الأعلى جريئًا؛ ويظهر الكمون المطلق أيضا.

### جوهر الاهتمام فقط

يبدأ بـ Q/K/V المعدة وينتهي عند مخرج الانتباه. لا يشمل إسقاط QKV، ومعيار Q/K، وRoPE، وإسقاط الإخراج، وAdaLN وFFN. يتضمن البوابات المتفرقة المطلوبة والتنبؤ والاختيار وإعادة الترتيب والترميم؛ يتضمن VDN كلاً من النوى الكاملة softmax والنوى الخطية. تستخدم كل طريقة هذه الحدود؛ لا يتم استبدال توقيت الوحدة بأكملها. أكمل التوقيت الأساسي 10/10 من أزواج الأجهزة المتطابقة؛ لا تقوم الخلايا المعلقة بإعادة استخدام قياسات الوحدة السابقة.

<div dir="ltr">

| إطارات | الطريقة | الكمون Ascend | الكمون برو 6000 | تسريع Ascend | تسريع برو 6000 |
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

### أكمل كتلة DiT

يتم تشغيله من كتلة الإدخال إلى كتلة الإخراج، بما في ذلك الإسقاطات والتطبيع وRoPE ونواة الانتباه وAdaLN وFFN والعمليات المتبقية. يتم قياسها بشكل مستقل عن القلب؛ الفرق بينهما ليس قياس FFN، ولا يتم استقراءه في وقت E2E.

<div dir="ltr">

| إطارات | الطريقة | الكمون Ascend | الكمون برو 6000 | تسريع Ascend | تسريع برو 6000 |
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

![BF16 مقارنة مجموعة الاهتمام DiT](../docs/figures/bf16-three-layer.png)[جميع السجلات ذات التشغيل التسعة والتحقق من هوية الإدخال](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1 / 2 / 4 / 8 بطاقات

1344×768، ثمانية تحديثات فعلية لـ DiT، بما في ذلك تشفير النص وفك تشفير الفيديو/الصوت والكتابة بتنسيق MP4. التحميل، يتم استبعاد الدمج السابق لـ LoRA، وإعداد AdaLN وعمليات الإحماء. تفضل المقيم DiT؛ تتحقق الصفوف المقيمة من عدم وجود أخطاء AdaLN ونقل الوزن في كل رتبة حول الطلب الرسمي. بعد OOMs المسجلة، تستخدم الخلايا ذات البطاقة الواحدة ذات 345 إطارًا المميزة مطابقة إلغاء التحميل وإعادة الاستخدام للطبقة الأصلية، والقياسات السابقة التي تم التحقق من صحتها بالكامل حسب الطلب؛ يتم وضع علامة واضحة على هذه الخلايا بدلاً من تقديمها كقياسات جديدة.

<div dir="ltr">

| إطارات | الطريقة | 1 بطاقة | 2 بطاقات | 4 بطاقات | 8 بطاقات |
| ---: | --- | ---: | ---: | ---: | ---: |
| 124 | Full | 197.66 s †T | 116.03 s | 64.78 s | 35.87 s |
| 124 | VSA V2 · 80% | 146.81 s †T | 93.57 s | 53.47 s | 30.40 s |
| 124 | Veda · 90% | 97.40 s †T | 69.66 s | 43.38 s | 25.33 s |
| 124 | VDN · official | 216.91 s †T | 136.02 s | 79.26 s | 44.36 s |
| 124 | Spark-H3 · 90% | 117.11 s †T | 79.41 s | 48.87 s | 29.05 s |
| 345 | Full | 1211.17 s †T | 649.72 s †T | 328.63 s | 178.81 s |
| 345 | VSA V2 · 80% | 617.55 s †L | 363.99 s †T | 193.33 s | 104.71 s |
| 345 | Veda · 90% | 337.95 s †T | 225.31 s †T | 130.11 s | 75.93 s |
| 345 | VDN · official | 620.67 s †L | 388.58 s †T | 209.84 s | 117.27 s |
| 345 | Spark-H3 · 90% | 419.44 s †L | 267.61 s †T | 150.50 s | 87.02 s |

</div>

مغطى **40/40**: 37 قياسًا جديدًا و3 قياسات سابقة مُعاد استخدامها. †T: إلغاء تحميل طبقة تشفير النص فقط؛ †L: إلغاء التحميل الأصلي لطبقة DiT/أداة تشفير النص باستخدام الجلب المسبق أحادي الطبقة وVAE المقيم. الخلايا غير المميزة ليس لها وقت تشغيل إلغاء التحميل. الصفوف المقيمة فقط هي التي تقوم بحساب AdaLN مسبقًا؛ لا تقدم صفوف إلغاء تحميل الطبقة أي مطالبة بنقل DiT أو نقل AdaLN صفر. VAE يصنف عدد البطاقات المطابقة؛ DiT رؤوس مقسمة 56/28/14/7. يتم الاحتفاظ بتغطية INT8 الأصلية.[سجلات INT8 E2E](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

يتم الاحتفاظ بمقارنة إلغاء تحميل الطبقة المطابقة السابقة في[مرجع البطاقة الفردية التاريخي](../docs/archive/matched-offload-single-card.md). ترتبط الخلايا المعاد استخدامها في الجدول الحالي بالسجلات الأصلية التي لم تتغير.

## النتائج النوعية والتحقق من الصحة

Ascend 910B3 فقط: **1344×768، 345 إطارًا أصليًا، 24 إطارًا في الثانية (14.375 ثانية، 14.4 ثانية تقريبًا)**، مع الصوت.

جميع الطرق الخمس تستخدم ثمانية مهاجمين. اليسار: BF16. على اليمين: W8A8 INT8 الأصلي. تستخدم جميع العينات بطاقة واحدة، ونفس الموجه، والبذرة المشتركة **7** والتفريغ المطابق.

> راكب دراجة يركب على طول مسار حديقة خضراء هادئة، ولقطة سينمائية سلسة، وضوء النهار الطبيعي.

<!-- BEGIN QUALITATIVE GALLERY -->
<table dir="ltr">
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**كامل + LightX2V8 · 8 خطوات**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**كامل + LightX2V8 · 8 خطوات**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 خطوات**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 خطوات**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 خطوات**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 خطوات**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · رسمي · 8 خطوات**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · رسمي · 8 خطوات**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 خطوات**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 خطوات**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

يستخدم VSA V2 80% من تناثر الفيديو؛ يستخدم Veda 90%. يحتفظ النص والصوت بقواعدهما الأصلية.[المعلمات، SHA256 والمصدر لجميع مقاطع الفيديو العشرة](../examples/videos/qualitative-1344-345.json). يتحقق فك التشفير الكامل من 345 إطارًا وصوتًا؛ تمت مراجعة 16 إطارًا موحدًا لكل مقطع بصريًا. لا يتم التأكيد على أن الطلاب المتميزين مكافئون للبكسل أو الإدراك الحسي. يحتفظ VSA بتغييرات اللقطة؛ لم يتم تقديم ضمان خالٍ من الوميض.

يستهدف التكامل الأولي دفعة واحدة من T2VA بدون مقاطع فيديو مرجعية. راجع[البنية](../docs/architecture.md) و[حالة التحقق](../docs/validation.md) و[إشعارات الطرف الثالث](../THIRD_PARTY_NOTICES.md).

</div>
