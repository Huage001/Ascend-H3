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

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | **Italiano** | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> **License:** Project-authored contributions in this distribution are offered under the [PolyForm Noncommercial License 1.0.0](../LICENSE). Third-party code and model weights have separate terms, and earlier Apache-2.0 grants remain in effect. See [third-party notices](../THIRD_PARTY_NOTICES.md). For uses not covered by the applicable terms, contact [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn).

> Questo README completo è una traduzione assistita dall'intelligenza artificiale del README inglese al commit be5c42e03f9a. Potrebbe diventare obsoleto; consultare la versione inglese per le informazioni più recenti. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

MiniMax Inferenza H3 su Ascend con attenzione completa, VSA, Veda, VDN e Spark-H3 in un repository. L'integrazione riutilizza il caricamento del modello vLLM-Omni, la quantizzazione del trasformatore INT8 e le teste di attenzione parallela del tensore nativo.

**Ambito convalidato:** 40/40 celle INT8 E2E a 1344×768, 124/345 fotogrammi e 1/2/4/8 NPU (37 nuove misurazioni e 3 celle storiche di layer-offload esplicitamente contrassegnate). Il confronto BF16 residente su Ascend 910B3 e PRO 6000 contiene 20 misurazioni del centro di attenzione e 20 misurazioni del blocco completo. Gli operatori di attenzione mantengono le loro versioni convalidate.

## Carichi di lavoro 768p predefiniti

| Caso | Larghezza × altezza | Frame di output nativi | Griglia video latente |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

Gli FPS di output sono 24. Le richieste utilizzano conteggi di frame espliciti, con l'interpolazione dei frame disabilitata. Il caso lungo è il carico di lavoro nativo concordato di 345 frame; non viene ritagliato a 341 fotogrammi. BF16 e INT8 utilizzano questi due carichi di lavoro 1344×768. Omettendo `--case` si seleziona `1344-124`.

<a id="install"></a>

## Installazione

Utilizza Linux, Python 3.12 e un ambiente Ascend Torch/CANN compatibile. Installare lo stack vLLM, vLLM-Ascend e vLLM-Omni bloccato descritto nelle [note sull'ambiente](../docs/environment.md). Non combinare versioni di framework non correlate. Quindi installa questo repository dal sorgente:

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

Le build non contengono pesi del modello e vengono eseguite senza occupare una NPU. Le librerie native vengono compilate localmente e conservate fuori dal checkout. Per un singolo backend, sostituire `all` con `vsa`, `veda` o `spark`; Completo e VDN non richiedono queste estensioni CANN personalizzate. Mantieni disponibile il checkout modificabile.

<a id="models"></a>

## Checkpoint e scelte dell'algoritmo

| Metodo | Asset del modello richiesti | Calcolo |
| --- | --- | --- |
| `full` | Base H3; LightX2V 8 passaggi per impostazione predefinita o community LoRA | Attenzione densa |
| `vsa` | Studente FastH3 V2 ufficiale completo, inclusi tutti i 50 gate di compressione addestrati | 8 avanti, scarsità video 80%; KV64 logico, blocchi KV selezionati accoppiati in KV128 sul chip |
| `veda` | Base H3 + LightX2V 8 passi + predittore Veda con addestramento ufficiale | Percorsi a blocchi dinamici pro capite Veda, scarsità nominale del 90%. |
| `vdn` | Base H3 + directory ufficiale del checkpoint stage-DMD VDN | Softmax locale più ramo DeltaNet appreso a 128 dimensioni |
| `spark` | Base H3 + LightX2V 8 gradini | Spark-H3 partizione gerarchica e softmax esatto/proxy congiunto, sparsità nominale del ramo esatto del 90% |

Scarica i modelli con le loro licenze originali. I pesi non vengono ridistribuiti. Diversi studenti distillati hanno orari diversi e regole apprese; spostare l'attenzione su pesi arbitrari non garantisce la corrispondenza della qualità. VSA richiede gate addestrati diversi da zero. VDN richiede i pesi dei rami appresi. I comandi di download e preparazione si trovano in [risorse modello](../docs/weights.md).

Completo, Veda e Spark abilitano il **LightX2V LoRA in otto passaggi bloccato per impostazione predefinita. Utilizzare `--no-few-step-lora` per eseguire il modello originale a molti passaggi. Una comunità preparata `--lora /models/style` viene fusa insieme a LightX2V prima che il caricamento termini; disabilitare lo studente predefinito quando si seleziona esplicitamente un altro studente della comunità. Veda ha sempre bisogno del suo predittore addestrato ufficiale. VSA accetta solo il checkpoint ufficiale completo **FastH3 V2 full**: otto avanti, scarsità video 80%, spostamenti video/audio 10/3, piastrella64, guida 1. Carica direttamente i pesi rilasciati e non accetta `--lora`. VDN accetta solo il checkpoint DMD ufficiale completo. Nessuno dei due accetta una comunità aggiuntiva LoRA o un flag di disabilitazione dello studente.

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

La fonte ufficiale PEFT viene normalizzata una volta utilizzando il suo alfa/rango dichiarato; i file preparati vengono sottoposti a controllo hash. Tutti gli adattatori selezionati si uniscono prima dell'affettamento TP e della quantizzazione INT8. Non sono presenti moltiplicazioni di matrici LoRA al momento della richiesta. Vedere [preparazione del modello](../docs/weights.md) per i formati sorgente e le pianificazioni fisse.

<a id="serve"></a>

## Avvia un server

Utilizza solo dispositivi che non hanno altri processi di elaborazione. Per una carta:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

Altri backend utilizzano la stessa API del server:

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

Esegui un server alla volta sui dispositivi selezionati. Aggiungi `--dry-run` per ispezionare il comando vLLM-Omni nativo generato senza caricare un modello.

Gli esempi a scheda singola trasmettono in streaming il codificatore di testo mantenendo residente INT8 DiT. La ricetta nativa quantizza DiT; il codificatore di testo rimane BF16, quindi la sua occupazione di memoria deve ancora essere inclusa nella politica di lancio. Per **BF16**, utilizzare `--precision bf16`; se necessario selezionare esplicitamente `--offload layer --no-precompute-adaln`. INT8 utilizza la ricetta nativa:

```json
{"transformer":{"method":"int8"}}
```

Le proiezioni INT8 alimentano l'attenzione di BF16 Q/K/V. Gli accumulatori di attenzione, le previsioni Veda e lo stato VDN mantengono la precisione prescritta. Questa è l'inferenza del modello INT8, non un'affermazione secondo cui ogni operazione di attenzione è INT8. La ricetta nativa mantiene i livelli non supportati in BF16. L'H3 a scheda singola ha una proiezione di 96.768 larghezze che supera il limite di uscita matmul quantizzato NPU di 65.535; TP può ridurre la larghezza per grado nell'intervallo.

Per **inferenza a quattro carte**, utilizzare lo sharding della testa parallela del tensore nativo:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices` supporta 1, 2, 4, 7 e 8, che dividono 56 teste. Veda conserva gli ID testa predittore globale dopo lo sharding. DiT TP a sette carte utilizza un rango di codificatore di testo perché le sue 64 query / teste da 8 KV non possono essere divise per sette. Avvia il valore predefinito su `--offload none` (pesi residenti). Se la memoria non è sufficiente, `--offload text` esegue lo streaming solo del codificatore di testo mantenendo DiT sul dispositivo; `--offload module` scambia componenti completi (qualificazione sperimentale); `--offload layer` trasmette in streaming entrambi i componenti, mantenendo un blocco attivo e uno precaricato. Il riferimento all'offload abbinato completato di seguito utilizza esplicitamente `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) su entrambe le piattaforme hardware. Le nuove misurazioni end-to-end utilizzano **solo INT8**, iniziando con l'esecuzione residente e contrassegnando ogni fallback Offload necessario. L'attenzione isolata e il confronto dei blocchi DiT utilizzano BF16 su entrambi i dispositivi: livelli 0/24/49, tre esecuzioni ciascuno, sommando i nove tempi e dividendo per nove. `--parallel ulysses` facoltativo è esposto per percorsi non VDN; TP rimane l'impostazione predefinita. La decodifica nativa VAE ora viene impostata per impostazione predefinita su tutte le carte selezionate (`--vae-parallel-size 0`); utilizzare `--vae-parallel-size 1` per un singolo rango di decodificatore. La pipeline INT8 Full/124 frame è stata superata con uno, due, quattro e otto ranghi VAE. Altre combinazioni metodo/conteggio fotogrammi rimangono nella matrice E2E.

I lanci della NPU residente preparano i risultati AdaLN prima che il server sia pronto. La preparazione risolve il modello caricato e unisce le posizioni sigma effettive di LoRA, il conteggio degli aggiornamenti di DiT e gli spostamenti video/audio; non è codificato in otto passaggi. Utilizzare `serve --steps N` per specificare un avvio insegnante/studente personalizzato e utilizzare lo stesso conteggio in `generate --steps N`. Una pianificazione distillata del checkpoint in conflitto viene rifiutata. La capacità della cache aumenta con i piani richiesti. La matematica della proiezione originale viene mantenuta; I pesi principali AdaLN rimangono sulla CPU, mentre i risultati preparati rimangono sulla NPU. TP4/TP8 memorizzano inoltre gli incorporamenti temporali nativi dagli input esatti del passo temporale per evitare che la variazione di riduzione FP32 a bit basso invalidi le cache downstream. Pesi, impostazioni numeriche e accordo sui risultati di tutti i ranghi continuano a tutelare il riutilizzo. Riavviare con la nuova configurazione per precalcolare una pianificazione diversa; un input non preparato utilizza il calcolo originale e può trasferire pesi. `--no-precompute-adaln` disabilita la preparazione; Lo streaming del modulo/livello DiT lo mantiene disabilitato per impostazione predefinita.

Per Spark/VSA/VDN a scheda singola da 345 frame, utilizzare `--offload layer --no-precompute-adaln` con gli operatori convalidati originali. La tabella E2E contrassegna queste celle di offload del livello nativo con †L. Tutte le altre celle mantengono la loro politica di residenza registrata individualmente.

Il programma di avvio imposta la scadenza per la richiesta del lavoratore nativo su 7200 secondi in modo che un video lungo non venga interrotto dall'impostazione predefinita di 600 secondi upstream. Imposta `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` prima del lancio per ignorare questo limite; cambia il comportamento di attesa, non il calcolo del modello o la durata misurata della richiesta.

## Genera i due casi 768p

Crea un file di prompt UTF-8, avvia un server, quindi utilizza il conteggio dei passi corrispondente:

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

Passa lo `--method` del server al client: tutti e cinque i metodi (Full/VSA/Veda/VDN/Spark) impostano per impostazione predefinita otto inoltri effettivi. Per il modello originale a più passaggi opzionale, utilizzare `generate --method full --no-few-step-lora` (50 passaggi), corrispondente al server. Un `--steps` esplicito rimane disponibile per uno studente della comunità preparato separatamente. `--guidance-scale` è opzionale; l’omissione preserva la politica nativa. Non viene aggiunta alcuna fase di perfezionamento. Il client salva l'MP4 e una ricevuta di temporizzazione JSON.

<!-- BEGIN BF16 LAYER RESULTS -->

## Confronto livelli BF16: residente su entrambi i dispositivi

Solo 1344×768, 124/345 fotogrammi. Pesi reali degli studenti e stati nascosti sintetici congelati identici a byte; queste non sono attivazioni di livelli acquisite da una richiesta video. I livelli 0, 24 e 49 ricevono ciascuno due riscaldamenti senza orario e tre corse formali: **somma i nove tempi e dividi per nove**. I pesi rimangono residenti su entrambi i dispositivi. Le attivazioni/pesi principali sono BF16; i predittori, gli accumuli e lo stato VDN FP32 richiesti vengono mantenuti. Gli aumenti di velocità utilizzano lo stesso hardware e la stessa forma BF16 Baseline completa. L'accelerazione relativa più elevata è in grassetto; viene mostrata anche la latenza assoluta.

### Solo centro di attenzione

Inizia con Q/K/V preparato e termina all'uscita dell'attenzione. Esclude proiezione QKV, norma Q/K, RoPE, proiezione di uscita, AdaLN e FFN. Include i necessari cancelli sparsi, previsione, selezione, riorganizzazione e restauro; VDN include sia nuclei softmax completi che lineari. Ogni metodo utilizza questo confine; i tempi dell'intero modulo non vengono sostituiti. Il timing del core ha completato 10/10 coppie hardware abbinate; le celle in sospeso non riutilizzano le misurazioni precedenti del modulo.

| Cornici | Metodo | Latenza Ascend | Latenza del PRO 6000 | Ascend accelerazione | PRO 6000 accelerazione |
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

### Blocco DiT completo

Va dall'input del blocco all'output del blocco, incluse proiezioni, normalizzazione, RoPE, nucleo di attenzione, AdaLN, FFN e operazioni residue. Misurato indipendentemente dal nucleo; la loro differenza non è una misurazione FFN e non viene nemmeno estrapolata nel tempo E2E.

| Cornici | Metodo | Latenza Ascend | Latenza del PRO 6000 | Ascend accelerazione | PRO 6000 accelerazione |
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

![BF16 nucleo di attenzione e confronto blocchi DiT](../docs/figures/bf16-three-layer.png)[Tutti i record di nove esecuzioni e i controlli di identità degli input](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1 / 2 / 4 / 8 carte

1344×768, otto aggiornamenti DiT effettivi, inclusa codifica di testo, decodifica video/audio e scrittura MP4. Sono esclusi il caricamento, l'unione precedente di LoRA, la preparazione di AdaLN e i riscaldamenti. Preferisco residente DiT; le righe residenti verificano zero AdaLN errori e trasferimenti di peso su ogni rango attorno alla richiesta formale. Dopo gli OOM registrati, le celle contrassegnate da 345 fotogrammi a scheda singola utilizzano l'offload del livello nativo e riutilizzano la corrispondenza, misurazioni precedenti completamente convalidate come richiesto; queste celle sono esplicitamente contrassegnate anziché presentate come nuove misurazioni.

| Cornici | Metodo | 1 carta | 2 carte | 4 carte | 8 carte |
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

Coperto **40/40**: 37 misurazioni precedenti fresche e 3 riutilizzate. †T: solo download del livello codificatore di testo; †L: Offload livello nativo DiT/codificatore di testo con prelettura a un livello e VAE residente. Le celle non contrassegnate non hanno alcun download di runtime. Solo le righe residenti precalcolano AdaLN; le righe layer-Offload non presentano alcuna richiesta di residenza DiT o di trasferimento zero-AdaLN. VAE classifica il conteggio delle carte della partita; DiT teste divise il 56/28/14/7. La copertura nativa INT8 viene mantenuta.[Registra INT8 E2E](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

Il precedente confronto layer-offload corrispondente viene mantenuto nel [riferimento storico a scheda singola](../docs/archive/matched-offload-single-card.md). Le celle riutilizzate nella tabella corrente si collegano ai record originali non modificati.

## Risultati qualitativi e validazione

Solo Ascend 910B3: **1344×768, 345 fotogrammi nativi, 24 FPS (14,375 secondi, circa 14,4 secondi)**, con audio.

Tutti e cinque i metodi utilizzano otto attaccanti. A sinistra: BF16. A destra: W8A8 nativo INT8. Tutti gli esempi utilizzano una scheda, lo stesso prompt, seed comune **7** e Offload abbinato.

> Un ciclista percorre un tranquillo sentiero verde del parco, una carrellata cinematografica fluida, luce naturale.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Completo + LightX2V8 · 8 passaggi**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Completo + LightX2V8 · 8 passaggi**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 passi**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 passi**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 passi**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 passi**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · ufficiale · 8 passaggi**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · ufficiale · 8 passaggi**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 passi**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 passi**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 utilizza una scarsità video dell'80%; Veda utilizza il 90%. Il testo e l'audio mantengono le regole native.[Parametri, SHA256 e provenienza per tutti e dieci i video](../examples/videos/qualitative-1344-345.json). La decodifica completa verifica 345 fotogrammi e audio; Sono stati esaminati visivamente 16 fotogrammi uniformi per clip. Non si afferma che gli studenti distinti siano pixel o percettivamente equivalenti. VSA mantiene i cambi di ripresa; non viene fornita alcuna garanzia senza sfarfallio.

L'integrazione iniziale è rivolta al batch-one T2VA senza video di riferimento. Vedere [architettura](../docs/architecture.md),[stato di convalida](../docs/validation.md) e [avvisi di terze parti](../THIRD_PARTY_NOTICES.md).
