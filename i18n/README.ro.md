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

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | **Română** | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> **License:** Project-authored contributions in this distribution are offered under the [PolyForm Noncommercial License 1.0.0](../LICENSE). Third-party code and model weights have separate terms, and earlier Apache-2.0 grants remain in effect. See [third-party notices](../THIRD_PARTY_NOTICES.md). For uses not covered by the applicable terms, contact [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn).

> Acest README complet este o traducere asistată de AI a README în engleză la commit be5c42e03f9a. Poate deveni învechit; consultați versiunea în limba engleză pentru cele mai recente informații. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

Inferența MiniMax H3 asupra Ascend cu atenție completă, VSA, Veda, VDN și Spark-H3 într-un singur depozit. Integrarea reutiliza încărcarea modelului vLLM-Omni, cuantizarea transformatorului INT8 și capetele de atenție paralele tensoare native.

**Domeniu validat:** 40/40 de celule INT8 E2E la 1344×768, 124/345 de cadre și 1/2/4/8 NPU (37 de măsurători noi și 3 celule istorice marcate explicit layer-Offload). Comparația rezidentă BF16 pe Ascend 910B3 și PRO 6000 conține 20 de măsurători de nucleu de atenție și 20 de măsurători bloc complet. Operatorii de atenție își păstrează versiunile validate.

## Sarcini de lucru implicite de 768p

| Caz | lățime × înălțime | Cadre native de ieșire | Grilă video latentă |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

FPS de ieșire este 24. Solicitările folosesc numărătoare explicite de cadre, cu interpolarea cadrelor dezactivată. Cazul lung este volumul de lucru nativ de 345 de cadre agreat; nu este decupat la 341 de cadre. BF16 și INT8 folosesc aceste două sarcini de lucru 1344×768. Omiterea `--case` selectează `1344-124`.

<a id="install"></a>

## Instalare

Utilizați Linux, Python 3.12 și un mediu compatibil Ascend Torch/CANN. Instalați stiva fixată vLLM, vLLM-Ascend și vLLM-Omni descrisă în [notele de mediu](../docs/environment.md). Nu combinați versiuni de cadru care nu au legătură. Apoi instalați acest depozit din sursă:

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

Construcțiile nu conțin greutăți de model și rulează fără a ocupa un NPU. Bibliotecile native sunt compilate local și păstrate în afara casei. Pentru un singur backend, înlocuiți `all` cu `vsa`, `veda` sau `spark`; Full și VDN nu necesită aceste extensii personalizate CANN. Păstrați disponibilul de casă editabil.

<a id="models"></a>

## Puncte de control și alegeri ale algoritmului

| Metoda | Elemente de model necesare | Calculul |
| --- | --- | --- |
| `full` | baza H3; LightX2V 8 pași în mod implicit sau o comunitate LoRA | Atenție densă |
| `vsa` | Student oficial complet FastH3 V2, inclusiv toate cele 50 de porți de compresie antrenate | 8 înainte, 80% video sparsity; KV64 logic, blocuri KV selectate împerecheate în KV128 pe cip |
| `veda` | Bază H3 + LightX2V în 8 pași + predictor Veda instruit oficial | Rute de blocare dinamice per cap Veda, dispersitate nominală de 90% |
| `vdn` | Baza H3 + directorul oficial al punctelor de control VDN stadiu-DMD | Softmax local plus ramura DeltaNet învățată cu 128 de dimensiuni |
| `spark` | Baza H3 + LightX2V 8 trepte | Spark-H3 partiție ierarhică și articulație exact/proxy softmax, 90% ramură exactă nominală |

Descărcați modele cu licențele lor originale. Greutățile nu sunt redistribuite. Diferiți elevi distilați au programe diferite și reguli învățate; schimbarea atenției asupra greutăților arbitrare nu garantează calitatea potrivirii. VSA necesită porți antrenate diferite de zero. VDN necesită greutățile sale de ramuri învățate. Comenzile de descărcare și pregătire sunt în [model assets](../docs/weights.md).

Complet, Veda și Spark activează **LightX2V LoRA** în opt pași fixat în mod implicit. Utilizați `--no-few-step-lora` pentru a rula modelul original cu mai mulți pași. O comunitate pregătită `--lora /models/style` este fuzionată împreună cu LightX2V înainte de finalizarea încărcării; dezactivați studentul implicit atunci când selectați în mod explicit un alt student din comunitate. Veda are întotdeauna nevoie de predictorul oficial antrenat. VSA acceptă numai punctul de control complet oficial **FastH3 V2 complet**: opt înainte, 80% dispersie video, deplasări video/audio 10/3, tile64, ghidare 1. Încarcă greutățile eliberate direct și nu acceptă `--lora`. VDN acceptă doar punctul său de control DMD de etapă complet. Niciunul nu acceptă o comunitate suplimentară LoRA sau un semnal de dezactivare a elevilor.

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

Sursa oficială PEFT este normalizată o dată folosind alfa/rangul declarat; fișierele pregătite sunt verificate prin hash. Toate adaptoarele selectate se îmbină înainte de tăierea TP și cuantizarea INT8. Nu există înmulțiri de matrice LoRA la momentul solicitării. Consultați [pregătirea modelului](../docs/weights.md) pentru formatele sursă și programele fixe.

<a id="serve"></a>

## Porniți un server

Utilizați numai dispozitive care nu au alte procese de calcul. Pentru un card:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

Alte backend-uri folosesc același server API:

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

Rulați câte un server pe dispozitivele selectate. Adăugați `--dry-run` pentru a inspecta comanda nativă vLLM-Omni generată fără a încărca un model.

Exemplele cu un singur card transmit codificatorul de text în timp ce păstrează rezidentul INT8 DiT. Rețeta nativă cuantifică DiT; codificatorul de text rămâne BF16, așa că amprenta sa de memorie trebuie să fie inclusă în politica de lansare. Pentru **BF16**, utilizați `--precision bf16`; dacă este necesar, selectați `--offload layer --no-precompute-adaln` în mod explicit. INT8 folosește rețeta nativă:

```json
{"transformer":{"method":"int8"}}
```

Proiecțiile INT8 pun în atenție BF16 Q/K/V. Acumulatorii de atenție, predicțiile Veda și starea VDN își păstrează precizia prescrisă. Aceasta este inferența modelului INT8, nu o pretenție că fiecare operație de atenție este INT8. Rețeta nativă păstrează straturi neacceptate în BF16. H3 cu un singur card are o proiecție lată de 96.768 care depășește limita de ieșire matmul cuantificată NPU de 65.535; TP își poate reduce lățimea pe rang în interval.

Pentru **inferența cu patru cărți**, utilizați fragmentarea capului paralel cu tensor nativ:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices` acceptă 1, 2, 4, 7 și 8, care împart 56 de capete. Veda păstrează ID-urile capului de predictor global după fragmentare. DiT TP cu șapte cărți folosește un rang de codificator de text, deoarece cele 64 de capete de interogare / 8 KV nu pot fi împărțite la șapte. Lansează implicit la `--offload none` (greutăți rezidente). Dacă memoria este insuficientă, `--offload text` transmite numai codificatorul de text, păstrând DiT pe dispozitiv; `--offload module` schimbă componente complete (calificare experimentală); `--offload layer` transmite în flux ambele componente, păstrând un bloc activ și unul preîncălcat. Referința completată de descărcare potrivită de mai jos utilizează în mod explicit `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) pe ambele platforme hardware. Noile măsurători end-to-end folosesc numai **INT8**, începând cu execuția rezidentă și marcând fiecare rezervă de descărcare necesară. Atenția izolată și comparația cu blocurile DiT utilizează BF16 pe ambele dispozitive: straturi 0/24/49, câte trei rulări, însumând cele nouă timpi și împărțind la nouă. Opțional `--parallel ulysses` este expus pentru căile non-VDN; TP rămâne implicit. Decodificarea nativă VAE este acum implicită pentru toate cardurile selectate (`--vae-parallel-size 0`); utilizați `--vae-parallel-size 1` pentru un singur rang de decodor. Conducta INT8 Full / 124-frame a trecut cu unul, doi, patru și opt rânduri VAE. Alte combinații de metodă/număr de cadre rămân în matricea E2E.

Lansările NPU rezidente pregătesc rezultatele AdaLN înainte de pregătirea serverului. Pregătirea rezolvă modelul încărcat și pozițiile sigma reale ale LoRA îmbinate, numărul de actualizări DiT și schimbările video/audio; nu este codificat în opt pași. Utilizați `serve --steps N` pentru a specifica o lansare profesor/elev personalizat și utilizați același număr în `generate --steps N`. Un program de control distilat conflictual este respins. Capacitatea memoriei cache crește odată cu planurile necesare. Matematica originală de proiecție este reținută; Greutățile principale AdaLN rămân pe CPU, în timp ce rezultatele pregătite rămân pe NPU. TP4/TP8 memorează, de asemenea, încorporarea timpului nativ din intrările exacte cu pas de timp pentru a evita variația de reducere a FP32 cu biți mici care invalidează memoria cache în aval. Greutățile, setările numerice și acordul de lovituri pentru toate rangurile asigură încă reutilizarea. Reporniți cu noua configurație pentru a precalcula un program diferit; o intrare nepregătită utilizează calculul original și poate transfera ponderi. `--no-precompute-adaln` dezactivează pregătirea; Modulul/stratul DiT îl menține dezactivat în mod implicit.

Pentru Spark cu un singur card cu 345 de cadre/VSA/VDN, utilizați `--offload layer --no-precompute-adaln` cu operatorii originali validați. Tabelul E2E marchează aceste celule de descărcare a stratului nativ cu †L; ‡R identifică măsurătorile care se potrivesc reutilizate din runda anterioară. Toate celelalte celule își păstrează politica de rezidență înregistrată individual.

Lansatorul setează termenul limită de solicitare a lucrătorului nativ la 7200 de secunde, astfel încât un videoclip lung să nu fie întrerupt de 600 de secunde implicite din amonte. Setați `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` înainte de lansare pentru a depăși această limită; modifică comportamentul de așteptare, nu calculul modelului sau durata măsurată a cererii.

## Generați cele două cazuri 768p

Creați un fișier prompt UTF-8, porniți un server, apoi utilizați numărul de pași corespunzători:

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

Transmiteți `--method` al serverului clientului: toate cele cinci metode (Full/VSA/Veda/VDN/Spark) implicit la opt redirecționări reale. Pentru modelul opțional original cu mai mulți pași, utilizați `generate --method full --no-few-step-lora` (50 de pași), potrivit cu serverul. Un `--steps` explicit rămâne disponibil pentru un student comunitar pregătit separat. `--guidance-scale` este opțional; omisiunea păstrează politica nativă. Nicio etapă de rafinare nu este adăugată. Clientul salvează MP4 și o chitanță de sincronizare JSON.

<!-- BEGIN BF16 LAYER RESULTS -->

## BF16 comparație strat: rezident pe ambele dispozitive

Numai 1344×768, 124/345 cadre. Greutăți reale ale elevilor și stări ascunse sintetice congelate identice pentru octeți; acestea nu sunt activări de strat capturate dintr-o solicitare video. Straturile 0, 24 și 49 primesc fiecare două încălziri necronometrate și trei runde formale: **suma cele nouă timpi și împarte la nouă**. Greutățile rămân rezidente pe ambele dispozitive. Activările/greutățile principale sunt BF16; predictorii FP32 necesari, acumulările și starea VDN sunt păstrate. Accelerările folosesc același hardware, aceeași formă BF16 Linia de bază completă. Accelerarea relativă mai mare este îndrăzneață; este afișată și latența absolută.

### Atenție numai de bază

Începe cu Q/K/V pregătit și se termină la ieșirea atenției. Exclude proiecția QKV, norma Q/K, RoPE, proiecția de ieșire, AdaLN și FFN. Include porți rare necesare, predicție, selecție, rearanjare și restaurare; VDN include atât miezuri softmax complete, cât și nuclee liniare. Fiecare metodă folosește această limită; cronometrajele întregului modul nu sunt înlocuite. Timingul de bază a finalizat 10/10 perechi hardware potrivite; celulele în așteptare nu reutilizați măsurătorile anterioare ale modulelor.

| Rame | Metoda | Ascend latență | Latență PRO 6000 | Ascend accelerare | Accelerarea PRO 6000 |
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

### Bloc complet DiT

Se execută de la intrarea blocului la ieșirea blocului, incluzând proiecții, normalizare, RoPE, nucleu de atenție, AdaLN, FFN și operațiuni reziduale. Măsurat independent de miez; diferența lor nu este o măsurătoare FFN și nici nu este extrapolată în timpul E2E.

| Rame | Metoda | Ascend latență | Latență PRO 6000 | Ascend accelerare | Accelerarea PRO 6000 |
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

![BF16 atenție nucleu și DiT comparație bloc](../docs/figures/bf16-three-layer.png)[Toate înregistrările din nouă rulări și verificările identității introduse](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1 / 2 / 4 / 8 cărți

1344×768, opt actualizări DiT, inclusiv codificarea textului, decodarea video/audio și scrierea MP4. Încărcarea, fuziunea anterioară LoRA, pregătirea AdaLN și încălzirile sunt excluse. Prefer rezident DiT; rândurile rezidente verifică zero AdaLN rateuri și transferuri de greutate pe fiecare rang în jurul cererii oficiale. După OOM înregistrate, celulele marcate cu un singur card cu 345 de cadre utilizează potrivirea de descărcare și reutilizare a stratului nativ, măsurători anterioare complet validate, conform solicitării; aceste celule sunt marcate în mod explicit, mai degrabă decât prezentate ca măsurători noi.

| Rame | Metoda | 1 card | 2 cărți | 4 cărți | 8 cărți |
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

Acoperit **40/40**: 37 măsurători anterioare proaspete și 3 refolosite. †T: strat codificator de text Numai descărcare; †L: DiT nativ/strat codificator de text Descărcare cu preîncărcare cu un singur strat și VAE rezident; ‡R: potrivirea măsurătorilor originale de la 2026-10-31. Celulele nemarcate nu au timp de rulare Descărcare. Numai rândurile rezidente precalculează AdaLN; layer-Rândurile de descărcare nu fac nicio revendicare DiT-reședință sau zero-AdaLN-transfer. VAE clasifică numărul de carduri de meci; DiT capete împărțite 56/28/14/7. Acoperirea nativă INT8 este păstrată.[INT8 Înregistrări E2E](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

Comparația layer-Descărcare potrivită anterioară este păstrată în [referința istorică pentru un singur card](../docs/archive/matched-offload-single-card.md). Celulele reutilizate din tabelul curent leagă la înregistrările originale neschimbate.

## Rezultate calitative și validare

Numai Ascend 910B3: **1344×768, 345 cadre native, 24 FPS (14,375 secunde, aproximativ 14,4 secunde)**, cu sunet.

Toate cele cinci metode folosesc opt atacanți. Stânga: BF16. Dreapta: nativ W8A8 INT8. Toate mostrele folosesc un singur card, același prompt, semințe comune **7** și descărcare potrivită.

> Un biciclist merge de-a lungul unei cărări liniștite în parc verde, filmare de urmărire cinematografică lină, lumină naturală.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Complet + LightX2V8 · 8 pași**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Complet + LightX2V8 · 8 pași**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 pași**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 pași**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 pași**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 pași**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · oficial · 8 pași**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · oficial · 8 pași**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 pași**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 pași**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 utilizează 80% dispersitate video; Veda utilizează 90%. Textul și audio își păstrează regulile native.[Parametri, SHA256 și proveniență pentru toate cele zece videoclipuri](../examples/videos/qualitative-1344-345.json). Decodificarea completă verifică 345 de cadre și sunet; Au fost revizuite vizual 16 rame uniforme per clip. Elevii diverși nu sunt afirmați a fi echivalenti în pixeli sau perceptuale. VSA păstrează modificările de lovituri; nu se face nicio garanție fără pâlpâire.

Integrarea inițială vizează lotul unu T2VA fără videoclipuri de referință. Consultați [arhitectura](../docs/architecture.md),[starea de validare](../docs/validation.md) și [notificări de la terți](../THIRD_PARTY_NOTICES.md).
