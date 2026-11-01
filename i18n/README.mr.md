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

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | **मराठी** | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> **License:** Project-authored contributions in this distribution are offered under the [PolyForm Noncommercial License 1.0.0](../LICENSE). Third-party code and model weights have separate terms, and earlier Apache-2.0 grants remain in effect. See [third-party notices](../THIRD_PARTY_NOTICES.md). For uses not covered by the applicable terms, contact [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn).

> हा संपूर्ण README कमिट be5c42e03f9a वर इंग्रजी README चे AI-सहाय्यित भाषांतर आहे. ते कालबाह्य होऊ शकते; नवीनतम माहितीसाठी इंग्रजी आवृत्ती पहा. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

MiniMax H3 अंदाजे Ascend वर पूर्ण, VSA, Veda, VDN आणि Spark-H3 लक्ष एका भांडारात. एकत्रीकरण vLLM-Omni मॉडेल लोडिंग, INT8 ट्रान्सफॉर्मर क्वांटायझेशन आणि नेटिव्ह टेन्सर पॅरलल अटेन्शन हेड्सचा पुन्हा वापर करते.

**प्रमाणित व्याप्ती:** 40/40 INT8 E2E सेल 1344×768, 124/345 फ्रेम्स आणि 1/2/4/8 NPUs (37 ताजे माप आणि 3 स्पष्टपणे चिन्हांकित ऐतिहासिक स्तर-ऑफलोड सेल). Ascend 910B3 आणि PRO 6000 वरील निवासी BF16 तुलनेमध्ये 20 लक्ष-कोर आणि 20 पूर्ण-ब्लॉक मोजमाप आहेत. लक्ष ऑपरेटर त्यांच्या प्रमाणित आवृत्त्या राखून ठेवतात.

## डीफॉल्ट 768p वर्कलोड्स

| केस | रुंदी × उंची | नेटिव्ह आउटपुट फ्रेम्स | सुप्त व्हिडिओ ग्रिड |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

आउटपुट FPS 24 आहे. फ्रेम इंटरपोलेशन अक्षम करून विनंत्या स्पष्ट फ्रेम संख्या वापरतात. लांब केस सहमत मूळ 345-फ्रेम वर्कलोड आहे; ते 341 फ्रेममध्ये क्रॉप केलेले नाही. BF16 आणि INT8 हे दोन 1344×768 वर्कलोड वापरतात. `--case` वगळणे `1344-124` निवडते.

<a id="install"></a>

## स्थापना

Linux, Python 3.12 आणि एक सुसंगत Ascend टॉर्च/CANN वातावरण वापरा. पिन केलेले vLLM, vLLM-Ascend आणि vLLM-Omni स्टॅक[पर्यावरण नोट्स](../docs/environment.md) मध्ये वर्णन केलेले स्थापित करा. असंबंधित फ्रेमवर्क रिलीझ एकत्र करू नका. नंतर स्त्रोतावरून हे भांडार स्थापित करा:

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

बिल्डमध्ये कोणतेही मॉडेल वजन नसतात आणि NPU व्यापल्याशिवाय चालतात. नेटिव्ह लायब्ररी स्थानिक पातळीवर संकलित केल्या जातात आणि चेकआउटच्या बाहेर ठेवल्या जातात. एका बॅकएंडसाठी, `all` च्या जागी `vsa`, `veda` किंवा `spark`; पूर्ण आणि VDN ला या सानुकूल CANN विस्तारांची आवश्यकता नाही. संपादन करण्यायोग्य चेकआउट उपलब्ध ठेवा.

<a id="models"></a>

## चेकपॉइंट्स आणि अल्गोरिदम निवडी

| पद्धत | आवश्यक मॉडेल मालमत्ता | गणना |
| --- | --- | --- |
| `full` | H3 बेस; LightX2V 8-चरण बाय डीफॉल्ट, किंवा समुदाय LoRA | दाट लक्ष |
| `vsa` | पूर्ण अधिकृत FastH3 V2 विद्यार्थी, सर्व 50 प्रशिक्षित कॉम्प्रेशन गेट्ससह | 8 फॉरवर्ड्स, 80% व्हिडिओ स्पेर्सिटी; लॉजिकल KV64, निवडलेले KV ब्लॉक्स चिपवर KV128 मध्ये जोडलेले |
| `veda` | H3 बेस + LightX2V 8-चरण + अधिकृत प्रशिक्षित Veda प्रेडिक्टर | डायनॅमिक प्रति-हेड Veda ब्लॉक मार्ग, नाममात्र 90% विरलता |
| `vdn` | H3 बेस + अधिकृत VDN स्टेज-DMD चेकपॉईंट निर्देशिका | स्थानिक सॉफ्टमॅक्स अधिक 128-आयामी शिकलेली डेल्टानेट शाखा |
| `spark` | H3 बेस + LightX2V 8-चरण | Spark-H3 श्रेणीबद्ध विभाजन आणि संयुक्त अचूक/प्रॉक्सी सॉफ्टमॅक्स, नाममात्र 90% अचूक-शाखा स्पेर्सिटी |

त्यांच्या मूळ परवान्याखाली मॉडेल डाउनलोड करा. वजनांचे पुनर्वितरण केले जात नाही. वेगवेगळ्या डिस्टिल्ड विद्यार्थ्यांचे वेळापत्रक वेगळे आणि शिकलेले नियम असतात; अनियंत्रित वजनांवर लक्ष वेधून घेतल्याने गुणवत्तेची हमी मिळत नाही. VSA ला शून्य प्रशिक्षित गेट्स आवश्यक आहेत. VDN ला त्याचे शिकलेले शाखा वजन आवश्यक आहे. डाउनलोड आणि तयारी आदेश[मॉडेल मालमत्ता](../docs/weights.md) मध्ये आहेत.

फुल, Veda आणि स्पार्क पिन केलेले **LightX2V आठ-चरण LoRA** बाय डीफॉल्ट सक्षम करतात. मूळ अनेक-चरण मॉडेल चालविण्यासाठी `--no-few-step-lora` वापरा. तयार केलेला समुदाय `--lora /models/style` लोडिंग पूर्ण होण्यापूर्वी LightX2V सह एकत्र केला जातो; स्पष्टपणे दुसरा समुदाय विद्यार्थी निवडताना डीफॉल्ट विद्यार्थी अक्षम करा. Veda नेहमी त्याच्या अधिकृत प्रशिक्षित प्रेडिक्टरची आवश्यकता असते. VSA फक्त पूर्ण अधिकृत **FastH3 V2 पूर्ण चेकपॉईंट** स्वीकारते: आठ फॉरवर्ड्स, 80% व्हिडिओ स्पेर्सिटी, व्हिडिओ/ऑडिओ शिफ्ट्स 10/3, टाइल64, मार्गदर्शन 1. ते सोडलेले वजन थेट लोड करते आणि `--lora` घेत नाही. VDN फक्त त्याचा संपूर्ण अधिकृत टप्पा-DMD चेकपॉइंट स्वीकारतो. कोणताही अतिरिक्त समुदाय LoRA किंवा विद्यार्थी-अक्षम ध्वज स्वीकारत नाही.

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

अधिकृत PEFT स्त्रोत एकदा घोषित अल्फा/रँक वापरून सामान्यीकृत केला जातो; तयार केलेल्या फाइल्स हॅश-चेक केल्या जातात. सर्व निवडलेले अडॅप्टर TP स्लाइसिंग आणि INT8 क्वांटायझेशनपूर्वी विलीन होतात. विनंती-वेळ LoRA मॅट्रिक्स गुणाकार नाहीत. स्रोत स्वरूप आणि निश्चित वेळापत्रकांसाठी[मॉडेल तयारी](../docs/weights.md) पहा.

<a id="serve"></a>

## सर्व्हर सुरू करा

इतर कोणतीही गणना प्रक्रिया नसलेली उपकरणेच वापरा. एका कार्डसाठी:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

इतर बॅकएंड समान सर्व्हर API वापरतात:

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

निवडलेल्या उपकरणांवर एका वेळी एक सर्व्हर चालवा. मॉडेल लोड न करता व्युत्पन्न केलेल्या vLLM-Omni कमांडची तपासणी करण्यासाठी `--dry-run` जोडा.

INT8 DiT निवासी ठेवताना सिंगल-कार्ड उदाहरणे मजकूर एन्कोडर प्रवाहित करतात. नेटिव्ह रेसिपी DiT ची मात्रा ठरवते; मजकूर एन्कोडर BF16 राहतो, त्यामुळे त्याची मेमरी फूटप्रिंट लाँच पॉलिसीमध्ये समाविष्ट करणे आवश्यक आहे. **BF16** साठी, `--precision bf16` वापरा; आवश्यक असल्यास स्पष्टपणे `--offload layer --no-precompute-adaln` निवडा. INT8 नेटिव्ह रेसिपी वापरते:

```json
{"transformer":{"method":"int8"}}
```

INT8 अंदाज BF16 Q/K/V लक्ष वेधून घेतात. लक्ष जमा करणारे, Veda अंदाज आणि VDN स्थिती त्यांची विहित अचूकता ठेवतात. हे INT8 मॉडेल अनुमान आहे, प्रत्येक लक्ष ऑपरेशन INT8 आहे असा दावा नाही. नेटिव्ह रेसिपी BF16 मध्ये असमर्थित स्तर राखून ठेवते. सिंगल-कार्ड H3 मध्ये 96,768-विस्तृत प्रोजेक्शन आहे जे 65,535 च्या NPU क्वांटाइज्ड मॅटमुल आउटपुट मर्यादा ओलांडते; TP त्याची प्रति-रँक रुंदी श्रेणीमध्ये कमी करू शकते.

**चार-कार्ड अनुमान** साठी, मूळ टेन्सर समांतर हेड शार्डिंग वापरा:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices` 1, 2, 4, 7 आणि 8 चे समर्थन करते, जे 56 हेड विभाजित करतात. Veda शार्डिंगनंतर ग्लोबल प्रेडिक्टर हेड आयडी राखून ठेवते. सात-कार्ड DiT TP एक मजकूर एन्कोडर रँक वापरतो कारण त्याची 64 क्वेरी / 8 KV हेड सात ने भागता येत नाही. `--offload none` (निवासी वजन) वर डीफॉल्ट लाँच करते. मेमरी अपुरी असल्यास, `--offload text` डिव्हाइसवर DiT राखून ठेवताना केवळ मजकूर एन्कोडर प्रवाहित करते; `--offload module` पूर्ण घटकांची अदलाबदल करते (प्रायोगिक पात्रता); `--offload layer` एक सक्रिय आणि एक प्रीफेच केलेला ब्लॉक ठेवून दोन्ही घटक प्रवाहित करते. खाली पूर्ण केलेला जुळलेला-ऑफलोड संदर्भ दोन्ही हार्डवेअर प्लॅटफॉर्मवर स्पष्टपणे `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) वापरतो. नवीन एन्ड-टू-एंड मोजमाप **INT8 फक्त** वापरतात, रहिवासी अंमलबजावणीपासून सुरुवात करून आणि प्रत्येक आवश्यक ऑफलोड फॉलबॅक चिन्हांकित करते. पृथक लक्ष आणि DiT-ब्लॉक तुलना दोन्ही उपकरणांवर BF16 वापरते: स्तर 0/24/49, प्रत्येकी तीन धावा, नऊ वेळेची बेरीज आणि नऊने भागणे. पर्यायी `--parallel ulysses` गैर-VDN पथांसाठी उघड आहे; TP डीफॉल्ट राहते. नेटिव्ह टाइल केलेले VAE डिकोडिंग आता सर्व निवडलेल्या कार्ड्सवर डीफॉल्ट आहे (`--vae-parallel-size 0`); सिंगल डीकोडर रँकसाठी `--vae-parallel-size 1` वापरा. INT8 फुल / 124-फ्रेम पाइपलाइन एक, दोन, चार आणि आठ VAE रँकसह पास झाली आहे. इतर पद्धत/फ्रेम-गणना संयोजन E2E मॅट्रिक्समध्ये राहतील.

रहिवासी NPU लाँच AdaLN निकाल सर्व्हरच्या तयारीपूर्वी तयार करतात. तयारी लोड केलेल्या मॉडेलचे निराकरण करते आणि LoRA च्या वास्तविक सिग्मा पोझिशन्स, DiT अद्यतन संख्या आणि व्हिडिओ/ऑडिओ शिफ्ट्स विलीन करते; ते आठ चरणांसाठी हार्ड-कोड केलेले नाही. शिक्षक/कस्टम-विद्यार्थी लाँच निर्दिष्ट करण्यासाठी `serve --steps N` वापरा आणि `generate --steps N` मध्ये समान संख्या वापरा. परस्परविरोधी डिस्टिल्ड चेकपॉईंट शेड्यूल नाकारले आहे. आवश्यक योजनांसह कॅशे क्षमता वाढते. मूळ प्रक्षेपणाचे गणित कायम ठेवले आहे; AdaLN मास्टर वेट CPU वर राहतात, तर तयार परिणाम NPU वर राहतात. TP4/TP8 कमी-बिट FP32 रिडक्शन व्हेरिएशन डाउनस्ट्रीम कॅशे अवैध ठरू नये म्हणून अचूक टाइमस्टेप इनपुटमधून नेटिव्ह टाइम एम्बेडिंग देखील लक्षात ठेवते. वजन, संख्यात्मक सेटिंग्ज आणि सर्व-रँक हिट करार अजूनही पुनर्वापराचे संरक्षण करतात. वेगळ्या वेळापत्रकाची पूर्वगणना करण्यासाठी नवीन कॉन्फिगरेशनसह रीस्टार्ट करा; अप्रस्तुत इनपुट मूळ गणनेचा वापर करते आणि वजन हस्तांतरित करू शकते. `--no-precompute-adaln` तयारी अक्षम करते; DiT मॉड्यूल/लेयर स्ट्रीमिंग ते डीफॉल्टनुसार अक्षम ठेवते.

सिंगल-कार्ड 345-फ्रेम स्पार्क/VSA/VDN साठी, मूळ प्रमाणित ऑपरेटरसह `--offload layer --no-precompute-adaln` वापरा. E2E सारणी या मूळ स्तर ऑफलोड पेशींना †L सह चिन्हांकित करते; ‡R मागील रनमधून पुन्हा वापरलेले जुळणारे मोजमाप ओळखते. इतर सर्व सेल त्यांचे वैयक्तिकरित्या रेकॉर्ड केलेले रेसिडेन्सी पॉलिसी राखून ठेवतात.

लाँचर नेटिव्ह वर्कर विनंतीची अंतिम मुदत 7200 सेकंदांवर सेट करते त्यामुळे अपस्ट्रीम 600-सेकंद डीफॉल्टद्वारे दीर्घ व्हिडिओ रद्द केला जात नाही. ही मर्यादा ओव्हरराइड करण्यासाठी लॉन्च करण्यापूर्वी `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` सेट करा; हे प्रतिक्षा वर्तन बदलते, मॉडेल गणना किंवा मोजलेली विनंती कालावधी नाही.

## दोन 768p केस व्युत्पन्न करा

UTF-8 प्रॉम्प्ट फाइल तयार करा, सर्व्हर सुरू करा, नंतर त्याची जुळणारी पायरी संख्या वापरा:

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

सर्व्हरचे `--method` क्लायंटला पास करा: सर्व पाच पद्धती (Full/VSA/Veda/VDN/Spark) आठ वास्तविक फॉरवर्डवर डीफॉल्ट. पर्यायी मूळ अनेक-चरण मॉडेलसाठी, सर्व्हरशी जुळणारे `generate --method full --no-few-step-lora` (50 चरणे) वापरा. स्वतंत्रपणे तयार केलेल्या समुदाय विद्यार्थ्यासाठी एक स्पष्ट `--steps` उपलब्ध आहे. `--guidance-scale` पर्यायी आहे; वगळणे मूळ धोरण जपते. कोणताही परिष्कृत टप्पा जोडलेला नाही. क्लायंट MP4 आणि JSON वेळेची पावती जतन करतो.

<!-- BEGIN BF16 LAYER RESULTS -->

## BF16 स्तर तुलना: दोन्ही डिव्हाइसेसवर निवासी

1344×768, 124/345 फ्रेम फक्त. वास्तविक विद्यार्थी वजन आणि बाइट-समान गोठविलेल्या सिंथेटिक लपलेल्या अवस्था; हे व्हिडिओ विनंतीवरून कॅप्चर केलेले स्तर सक्रियकरण नाहीत. स्तर 0, 24 आणि 49 प्रत्येकी दोन वेळेशिवाय वार्मअप आणि तीन औपचारिक धावा मिळतात: **नऊ वेळेची बेरीज करा आणि नऊने भागा**. वजन दोन्ही उपकरणांवर राहतात. सक्रियता/मुख्य वजने BF16 आहेत; आवश्यक FP32 अंदाज, संचय आणि VDN स्थिती राखून ठेवली आहे. स्पीडअप्स समान-हार्डवेअर, समान-आकार BF16 पूर्ण बेसलाइन वापरतात. उच्च सापेक्ष गती ठळक आहे; परिपूर्ण विलंब देखील दर्शविला आहे.

### लक्ष कोर फक्त

तयार Q/K/V ने सुरू होते आणि लक्ष आउटपुटवर समाप्त होते. QKV प्रोजेक्शन, Q/K नॉर्म, RoPE, आउटपुट प्रोजेक्शन, AdaLN आणि FFN वगळले आहे. आवश्यक विरळ गेट्स, अंदाज, निवड, पुनर्रचना आणि जीर्णोद्धार समाविष्ट आहे; VDN मध्ये संपूर्ण सॉफ्टमॅक्स आणि रेखीय कोर दोन्ही समाविष्ट आहेत. प्रत्येक पद्धत ही सीमा वापरते; संपूर्ण-मॉड्यूल वेळा बदलल्या जात नाहीत. कोर टाइमिंगने 10/10 जुळलेल्या हार्डवेअर जोड्या पूर्ण केल्या आहेत; प्रलंबित सेल पूर्वीचे मॉड्यूल मोजमाप पुन्हा वापरत नाहीत.

| फ्रेम्स | पद्धत | Ascend विलंबता | PRO 6000 विलंब | Ascend स्पीडअप | प्रो 6000 स्पीडअप |
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

### DiT ब्लॉक पूर्ण करा

प्रक्षेपण, सामान्यीकरण, RoPE, अटेन्शन कोअर, AdaLN, FFN आणि अवशिष्ट ऑपरेशन्ससह ब्लॉक इनपुटपासून ब्लॉक आउटपुटपर्यंत चालते. कोरपासून स्वतंत्रपणे मोजले जाते; त्यांच्यातील फरक हे FFN मापन नाही आणि E2E वेळेत एक्स्ट्रापोलेट केलेले नाही.

| फ्रेम्स | पद्धत | Ascend विलंबता | PRO 6000 विलंब | Ascend स्पीडअप | प्रो 6000 स्पीडअप |
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

![BF16 अटेन्शन कोअर आणि DiT ब्लॉक तुलना](../docs/figures/bf16-three-layer.png)[सर्व नऊ-रन रेकॉर्ड आणि इनपुट ओळख तपासणी](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1 / 2 / 4 / 8 कार्ड

1344×768, आठ वास्तविक DiT अद्यतने, मजकूर एन्कोडिंग, व्हिडिओ/ऑडिओ डीकोडिंग आणि MP4 लेखन. लोड होत आहे, पूर्वीचे LoRA विलीनीकरण, AdaLN तयारी आणि वॉर्मअप वगळले आहेत. रहिवासी DiT पसंत करा; रहिवासी पंक्ती औपचारिक विनंतीच्या आसपास प्रत्येक रँकवर शून्य AdaLN चुकते आणि वजन हस्तांतरण सत्यापित करतात. रेकॉर्ड केलेल्या OOMs नंतर, चिन्हांकित सिंगल-कार्ड 345-फ्रेम सेल नेटिव्ह लेयर ऑफलोड वापरतात आणि मॅचिंगचा पुनर्वापर करतात, विनंती केल्यानुसार आधीच्या मोजमापांची पूर्णपणे वैधता; या पेशी ताजे मोजमाप म्हणून सादर करण्याऐवजी स्पष्टपणे चिन्हांकित केल्या आहेत.

| फ्रेम्स | पद्धत | 1 कार्ड | 2 कार्डे | 4 कार्डे | 8 कार्डे |
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

कव्हर केलेले **40/40**: 37 ताजे आणि 3 पुन्हा वापरलेले पूर्वीचे मोजमाप. †T: मजकूर-एनकोडर स्तर केवळ ऑफलोड; †L: मूळ DiT/टेक्स्ट-एनकोडर स्तर ऑफलोड एक-लेयर प्रीफेच आणि निवासी VAE; ‡R: 2026-10-31 कडील मूळ मोजमापांशी जुळणारे. चिन्हांकित नसलेल्या सेलमध्ये रनटाइम ऑफलोड नाही. केवळ निवासी पंक्ती AdaLN प्रीकॉम्प्युट करतात; स्तर-ऑफलोड पंक्ती DiT-रेसिडेन्सी किंवा शून्य-AdaLN-हस्तांतरण दावा करत नाहीत. VAE रँक मॅच कार्ड संख्या; DiT हेडस् स्प्लिट 56/28/14/7. नेटिव्ह INT8 कव्हरेज राखून ठेवले आहे.[INT8 E2E रेकॉर्ड](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

पूर्वीची जुळलेली लेयर-ऑफलोड तुलना[ऐतिहासिक एकल-कार्ड संदर्भ](../docs/archive/matched-offload-single-card.md) मध्ये ठेवली आहे. वर्तमान सारणीतील पुनर्वापरलेले सेल अपरिवर्तित मूळ रेकॉर्डशी लिंक करतात.

## गुणात्मक परिणाम आणि प्रमाणीकरण

Ascend 910B3 फक्त: **1344×768, 345 नेटिव्ह फ्रेम, 24 FPS (14.375 सेकंद, अंदाजे 14.4 सेकंद)**, ऑडिओसह.

सर्व पाच पद्धती आठ फॉरवर्ड वापरतात. डावीकडे: BF16. उजवीकडे: मूळ W8A8 INT8. सर्व नमुने एक कार्ड, समान प्रॉम्प्ट, सामान्य बियाणे **7** आणि जुळलेले ऑफलोड वापरतात.

> एक सायकलस्वार शांत हिरव्या पार्क मार्गावर, गुळगुळीत सिनेमॅटिक ट्रॅकिंग शॉट, नैसर्गिक दिवसाचा प्रकाश.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**पूर्ण + LightX2V8 · ८ पायऱ्या**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**पूर्ण + LightX2V8 · ८ पायऱ्या**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 चरण**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 चरण**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 पायऱ्या**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 पायऱ्या**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · अधिकृत · ८ पायऱ्या**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · अधिकृत · ८ पायऱ्या**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · ८ पायऱ्या**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · ८ पायऱ्या**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 80% व्हिडिओ स्पेर्सिटी वापरते; Veda 90% वापरते. मजकूर आणि ऑडिओ त्यांचे मूळ नियम राखून ठेवतात.[पॅरामीटर्स, SHA256 आणि सर्व दहा व्हिडिओंसाठी मूळ](../examples/videos/qualitative-1344-345.json). पूर्ण डीकोडिंग 345 फ्रेम आणि ऑडिओ सत्यापित करते; प्रति क्लिप 16 एकसमान फ्रेमचे दृष्यदृष्ट्या पुनरावलोकन केले गेले. वेगळे विद्यार्थी पिक्सेल- किंवा आकलनदृष्ट्या समतुल्य असल्याचे प्रतिपादन केले जात नाही. VSA शॉट बदल राखून ठेवते; फ्लिकर-फ्री हमी दिलेली नाही.

प्रारंभिक एकीकरण संदर्भ व्हिडिओशिवाय बॅच-वन T2VA ला लक्ष्य करते. पहा[आर्किटेक्चर](../docs/architecture.md),[प्रमाणीकरण स्थिती](../docs/validation.md) आणि[तृतीय-पक्ष सूचना](../THIRD_PARTY_NOTICES.md).
