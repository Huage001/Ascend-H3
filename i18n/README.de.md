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

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | **Deutsch** | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> **License:** Project-authored contributions in this distribution are offered under the [PolyForm Noncommercial License 1.0.0](../LICENSE). Third-party code and model weights have separate terms, and earlier Apache-2.0 grants remain in effect. See [third-party notices](../THIRD_PARTY_NOTICES.md). For uses not covered by the applicable terms, contact [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn).

> Diese vollständige README-Datei ist eine KI-gestützte Übersetzung der englischen README-Datei unter Commit be5c42e03f9a. Es kann veraltet sein; Aktuelle Informationen finden Sie in der englischen Version. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

MiniMax H3-Inferenz auf Ascend mit voller Aufmerksamkeit, VSA, Veda, VDN und Spark-H3 in einem Repository. Die Integration verwendet das Laden des vLLM-Omni-Modells, die Transformatorquantisierung INT8 und native Tensor-Parallel-Aufmerksamkeitsköpfe wieder.

**Validierter Umfang:** 40/40 INT8 E2E-Zellen bei 1344×768, 124/345 Frames und 1/2/4/8 NPUs (37 neue Messungen und 3 explizit markierte historische Layer-Offload-Zellen). Der residente BF16-Vergleich auf Ascend 910B3 und PRO 6000 enthält 20 Aufmerksamkeitskern- und 20 Komplettblockmessungen. Die Aufmerksamkeitsoperatoren behalten ihre validierten Versionen.

## Standardmäßige 768p-Workloads

| Fall | Breite × Höhe | Native Ausgaberahmen | Latentes Videoraster |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

Die Ausgabe-FPS beträgt 24. Anfragen verwenden explizite Frame-Zählungen, wobei die Frame-Interpolation deaktiviert ist. Der lange Fall ist die vereinbarte native 345-Frame-Arbeitslast; es ist nicht auf 341 Bilder zugeschnitten. BF16 und INT8 verwenden diese beiden 1344×768-Workloads. Wenn `--case` weggelassen wird, wird `1344-124` ausgewählt.

<a id="install"></a>

## Installation

Verwenden Sie Linux, Python 3.12 und eine kompatible Ascend Torch/CANN-Umgebung. Installieren Sie den angehefteten vLLM-, vLLM-Ascend- und vLLM-Omni-Stack, der in [den Umgebungshinweisen](../docs/environment.md) beschrieben wird. Kombinieren Sie keine unabhängigen Framework-Releases. Dann installieren Sie dieses Repository aus der Quelle:

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

Builds enthalten keine Modellgewichte und werden ausgeführt, ohne eine NPU zu belegen. Native Bibliotheken werden lokal kompiliert und außerhalb der Kasse aufbewahrt. Ersetzen Sie für ein einzelnes Backend `all` durch `vsa`, `veda` oder `spark`. Für Full und VDN sind diese benutzerdefinierten CANN-Erweiterungen nicht erforderlich. Halten Sie den bearbeitbaren Checkout verfügbar.

<a id="models"></a>

## Prüfpunkte und Algorithmusauswahl

| Methode | Erforderliche Modellressourcen | Berechnung |
| --- | --- | --- |
| `full` | H3-Basis; LightX2V 8-stufig standardmäßig oder eine Community LoRA | Dichte Aufmerksamkeit |
| `vsa` | Kompletter offizieller FastH3 V2-Student, einschließlich aller 50 trainierten Kompressionsgates | 8 Vorwärtsbewegungen, 80 % Videosparsity; logischer KV64, ausgewählte KV-Blöcke gepaart mit KV128 auf dem Chip |
| `veda` | H3-Basis + LightX2V 8-Stufen + offiziell geschulter Veda-Prädiktor | Dynamische Veda-Blockrouten pro Kopf, nominal 90 % Sparsity |
| `vdn` | H3-Basis + offizielles VDN Stage-DMD-Checkpoint-Verzeichnis | Lokaler Softmax plus 128-dimensionaler erlernter DeltaNet-Zweig |
| `spark` | H3-Basis + LightX2V 8-stufig | Spark-H3 hierarchische Partition und gemeinsamer exakter/Proxy-Softmax, nominal 90 % exakte Zweigsparsität |

Laden Sie Modelle unter ihren Originallizenzen herunter. Gewichte werden nicht umverteilt. Unterschiedlich destillierte Schüler haben unterschiedliche Zeitpläne und erlernte Regeln; Das Vertauschen der Aufmerksamkeit auf beliebige Gewichtungen garantiert keine Übereinstimmungsqualität. VSA erfordert trainierte Gates ungleich Null. VDN benötigt die erlernten Zweiggewichte. Download- und Vorbereitungsbefehle finden Sie in [Modellressourcen](../docs/weights.md).

Full, Veda und Spark aktivieren standardmäßig den angehefteten **LightX2V achtstufigen LoRA**. Verwenden Sie `--no-few-step-lora`, um das ursprüngliche mehrstufige Modell auszuführen. Eine vorbereitete Community `--lora /models/style` wird mit LightX2V zusammengeführt, bevor der Ladevorgang abgeschlossen ist. Deaktivieren Sie den Standardschüler, wenn Sie explizit einen anderen Community-Studenten auswählen. Veda benötigt immer seinen offiziell geschulten Prädiktor. VSA akzeptiert nur den vollständigen offiziellen **FastH3 V2-Vollprüfpunkt**: acht Vorwärtsbewegungen, 80 % Videosparsity, Video-/Audioverschiebungen 10/3, Tile64, Anleitung 1. Es lädt die freigegebenen Gewichte direkt und übernimmt `--lora` nicht. VDN akzeptiert nur den vollständigen offiziellen Stage-DMD-Checkpoint. Keiner von beiden akzeptiert eine zusätzliche Community-LoRA- oder Student-Disable-Flagge.

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

Die offizielle PEFT-Quelle wird einmal mit ihrem deklarierten Alpha/Rang normalisiert; Vorbereitete Dateien werden einem Hash-Check unterzogen. Alle ausgewählten Adapter werden vor dem TP-Slicing und der INT8-Quantisierung zusammengeführt. Es gibt keine LoRA-Matrixmultiplikationen zur Anforderungszeit. Informationen zu Quellformaten und festen Zeitplänen finden Sie unter [Modellvorbereitung](../docs/weights.md).

<a id="serve"></a>

## Starten Sie einen Server

Verwenden Sie nur Geräte, die über keine anderen Rechenprozesse verfügen. Für eine Karte:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

Andere Backends verwenden dieselbe Server-API:

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

Führen Sie jeweils einen Server auf den ausgewählten Geräten aus. Fügen Sie `--dry-run` hinzu, um den generierten nativen Befehl vLLM-Omni zu überprüfen, ohne ein Modell zu laden.

Die Einzelkartenbeispiele streamen den Text-Encoder, während INT8 DiT resident bleibt. Das native Rezept quantisiert DiT; Der Textencoder bleibt BF16, daher muss sein Speicherbedarf weiterhin in der Startrichtlinie berücksichtigt werden. Für **BF16** verwenden Sie `--precision bf16`; Wählen Sie ggf. explizit `--offload layer --no-precompute-adaln` aus. INT8 verwendet das native Rezept:

```json
{"transformer":{"method":"int8"}}
```

INT8-Projektionen lenken die Aufmerksamkeit auf BF16 Q/K/V. Aufmerksamkeitsakkumulatoren, Veda-Vorhersagen und VDN-Zustand behalten ihre vorgeschriebene Präzision. Dies ist eine INT8-Modellinferenz und keine Behauptung, dass jede Aufmerksamkeitsoperation INT8 ist. Das native Rezept behält nicht unterstützte Ebenen in BF16 bei. Die Einzelkarte H3 verfügt über eine 96.768-breite Projektion, die die NPU-quantisierte Matmul-Ausgabegrenze von 65.535 überschreitet; TP kann seine Breite pro Rang in die Reichweite reduzieren.

Für **Vier-Karten-Inferenz** verwenden Sie natives Tensor-Parallel-Head-Sharding:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices` unterstützt 1, 2, 4, 7 und 8, die 56 Köpfe teilen. Veda behält globale Prädiktorkopf-IDs nach dem Sharding bei. DiT TP mit sieben Karten verwendet einen Text-Encoder-Rang, da seine 64 Abfrage-/8 KV-Köpfe nicht durch sieben geteilt werden können. Startet standardmäßig `--offload none` (residente Gewichte). Wenn der Speicher nicht ausreicht, streamt `--offload text` nur den Textencoder, während DiT auf dem Gerät beibehalten wird; `--offload module` tauscht komplette Komponenten aus (experimentelle Qualifikation); `--offload layer` streamt beide Komponenten und behält einen aktiven und einen vorab abgerufenen Block bei. Die vollständige Matched-Offload-Referenz unten verwendet explizit `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) auf beiden Hardwareplattformen. Neue End-to-End-Messungen verwenden nur **INT8**, beginnend mit der residenten Ausführung und der Markierung aller erforderlichen Offload-Fallbacks. Der isolierte Aufmerksamkeits- und DiT-Blockvergleich verwendet BF16 auf beiden Geräten: Schichten 0/24/49, jeweils drei Läufe, Summierung der neun Timings und Division durch neun. Optional ist `--parallel ulysses` für Nicht-VDN-Pfade verfügbar; TP bleibt die Standardeinstellung. Die native gekachelte VAE-Dekodierung gilt jetzt standardmäßig für alle ausgewählten Karten (`--vae-parallel-size 0`); Verwenden Sie `--vae-parallel-size 1` für einen einzelnen Decoderrang. Die INT8 Full / 124-Frame-Pipeline wurde mit einem, zwei, vier und acht VAE-Rängen bestanden. Andere Methoden-/Frameanzahl-Kombinationen verbleiben in der E2E-Matrix.

Residente NPU-Starts bereiten AdaLN-Ergebnisse vor, bevor der Server bereit ist. Durch die Vorbereitung wird das geladene Modell aufgelöst und die tatsächlichen Sigma-Positionen von LoRA, die Aktualisierungsanzahl von DiT und die Video-/Audioverschiebungen zusammengeführt. Es ist nicht auf acht Schritte fest codiert. Verwenden Sie `serve --steps N`, um einen Lehrer-/benutzerdefinierten Schülerstart anzugeben, und verwenden Sie dieselbe Anzahl in `generate --steps N`. Ein widersprüchlicher destillierter Prüfpunktplan wird abgelehnt. Die Cache-Kapazität wächst mit den erforderlichen Plänen. Die ursprüngliche Projektionsberechnung bleibt erhalten; AdaLN Mastergewichtungen bleiben auf der CPU, während vorbereitete Ergebnisse auf der NPU bleiben. TP4/TP8 speichert auch native Zeiteinbettungen aus exakten Zeitschritteingaben, um zu vermeiden, dass Low-Bit-FP32-Reduktionsschwankungen Downstream-Caches ungültig machen. Gewichtungen, numerische Einstellungen und die Trefferübereinstimmung aller Ränge schützen weiterhin die Wiederverwendung. Starten Sie mit der neuen Konfiguration neu, um einen anderen Zeitplan vorab zu berechnen. Eine unvorbereitete Eingabe verwendet die ursprüngliche Berechnung und kann Gewichtungen übertragen. `--no-precompute-adaln` deaktiviert die Vorbereitung; Beim DiT-Modul-/Layer-Streaming bleibt es standardmäßig deaktiviert.

Für Spark/VSA/VDN mit einer Karte und 345 Frames verwenden Sie `--offload layer --no-precompute-adaln` mit den ursprünglich validierten Operatoren. Die E2E-Tabelle markiert diese Offload-Zellen der nativen Schicht mit †L. Alle anderen Zellen behalten ihre individuell erfasste Aufenthaltsrichtlinie.

Der Launcher legt die Deadline für native Worker-Anfragen auf 7200 Sekunden fest, sodass ein langes Video nicht durch die Upstream-Standardeinstellung von 600 Sekunden abgebrochen wird. Legen Sie `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` vor dem Start fest, um dieses Limit zu überschreiben. Es ändert das Warteverhalten, nicht die Modellberechnung oder die gemessene Anforderungsdauer.

## Generieren Sie die beiden 768p-Fälle

Erstellen Sie eine UTF-8-Eingabeaufforderungsdatei, starten Sie einen Server und verwenden Sie dann die entsprechende Schrittanzahl:

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

Übergeben Sie den `--method` des Servers an den Client: Alle fünf Methoden (Full/VSA/Veda/VDN/Spark) sind standardmäßig auf acht tatsächliche Weiterleitungen eingestellt. Für das optionale Originalmodell mit vielen Schritten verwenden Sie `generate --method full --no-few-step-lora` (50 Schritte), passend zum Server. Für einen separat vorbereiteten Community-Studenten bleibt ein expliziter `--steps` verfügbar. `--guidance-scale` ist optional; Durch Auslassung bleibt die einheimische Politik erhalten. Es wird keine Verfeinerungsphase hinzugefügt. Der Client speichert die MP4- und eine JSON-Timing-Quittung.

<!-- BEGIN BF16 LAYER RESULTS -->

## BF16-Schichtenvergleich: resident auf beiden Geräten

1344×768, nur 124/345 Bilder. Echte Studentengewichte und byteidentische eingefrorene synthetische verborgene Zustände; Hierbei handelt es sich nicht um Layer-Aktivierungen, die aus einer Videoanfrage erfasst wurden. Die Schichten 0, 24 und 49 erhalten jeweils zwei Aufwärmübungen ohne Zeitangabe und drei formelle Läufe: **Summiere die neun Zeiten und dividiere durch neun**. Die Gewichte bleiben auf beiden Geräten gespeichert. Aktivierungen/Hauptgewichte sind BF16; Erforderliche FP32-Prädiktoren, Akkumulationen und der VDN-Status bleiben erhalten. Beschleunigungen verwenden die gleiche Hardware und die gleiche Form. BF16 Vollständige Basislinie. Die höhere relative Beschleunigung ist fett gedruckt; Es wird auch die absolute Latenz angezeigt.

### Nur Aufmerksamkeitskern

Beginnt mit vorbereitetem Q/K/V und endet mit der Aufmerksamkeitsausgabe. Ausgeschlossen sind QKV-Projektion, Q/K-Norm, RoPE, Ausgabeprojektion, AdaLN und FFN. Beinhaltet erforderliche Sparse-Gates, Vorhersage, Auswahl, Neuanordnung und Wiederherstellung; VDN umfasst sowohl vollständige Softmax- als auch lineare Kerne. Jede Methode verwendet diese Grenze; Die Zeiten für das gesamte Modul werden nicht ersetzt. Das Kerntiming hat 10/10 übereinstimmende Hardwarepaare abgeschlossen; Ausstehende Zellen verwenden frühere Modulmessungen nicht wieder.

| Rahmen | Methode | Ascend Latenz | PRO 6000 Latenz | Ascend Beschleunigung | PRO 6000-Beschleunigung |
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

### Kompletter DiT-Block

Läuft von der Blockeingabe zur Blockausgabe, einschließlich Projektionen, Normalisierung, RoPE, Aufmerksamkeitskern, AdaLN, FFN und Restoperationen. Unabhängig vom Kern gemessen; Ihr Unterschied ist keine FFN-Messung und wird auch nicht in die E2E-Zeit extrapoliert.

| Rahmen | Methode | Ascend Latenz | PRO 6000 Latenz | Ascend Beschleunigung | PRO 6000-Beschleunigung |
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

![BF16 Aufmerksamkeitskern und DiT Blockvergleich](../docs/figures/bf16-three-layer.png)[Alle Neun-Run-Datensätze und eingegebenen Identitätsprüfungen](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1 / 2 / 4 / 8 Karten

1344×768, acht aktuelle DiT-Updates, einschließlich Textkodierung, Video-/Audio-Dekodierung und MP4-Schreiben. Laden, vorheriges LoRA-Merge, AdaLN-Vorbereitung und Aufwärmen sind ausgeschlossen. Resident DiT bevorzugen; Residente Zeilen verifizieren null AdaLN Fehler und Gewichtsübertragungen auf jeder Rangstufe rund um die formelle Anfrage. Nach der Aufzeichnung von OOMs verwenden die markierten Einzelkarten-345-Frame-Zellen natives Layer-Offload und Wiederverwendungs-Matching sowie vollständig validierte vorherige Messungen wie angefordert. Diese Zellen werden explizit markiert und nicht als neue Messungen dargestellt.

| Rahmen | Methode | 1 Karte | 2 Karten | 4 Karten | 8 Karten |
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

Abgedeckt **40/40**: 37 frische und 3 wiederverwendete vorherige Messungen. †T: Nur Text-Encoder-Layer-Offload; †L: natives DiT/Text-Encoder-Layer-Offload mit einschichtigem Prefetch und residentem VAE. Für nicht markierte Zellen gibt es kein Laufzeit-Offload. Nur residente Zeilen berechnen AdaLN vorab; Layer-Offload-Zeilen erheben keinen Anspruch auf DiT-Residency oder Null-AdaLN-Transfer. VAE-Ränge entsprechen der Kartenanzahl; DiT Köpfe geteilt 56/28/14/7. Die native INT8-Abdeckung bleibt erhalten.[INT8 E2E-Datensätze](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

Der zuvor übereinstimmende Layer-Offload-Vergleich bleibt in der [historischen Einzelkartenreferenz](../docs/archive/matched-offload-single-card.md) erhalten. Wiederverwendete Zellen in der aktuellen Tabelle verweisen auf unveränderte Originaldatensätze.

## Qualitative Ergebnisse und Validierung

Ascend Nur 910B3: **1344×768, 345 native Bilder, 24 FPS (14,375 Sekunden, ca. 14,4 Sekunden)**, mit Audio.

Alle fünf Methoden verwenden acht Stürmer. Links: BF16. Rechts: natives W8A8 INT8. Alle Beispiele verwenden eine Karte, denselben Prompt, einen gemeinsamen Seed **7** und einen passenden Offload.

> Ein Radfahrer fährt einen ruhigen grünen Parkweg entlang, sanfte filmische Kamerafahrt, natürliches Tageslicht.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Voll + LightX2V8 · 8 Schritte**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Voll + LightX2V8 · 8 Schritte**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80 % · 8 Schritte**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80 % · 8 Schritte**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90 % · 8 Schritte**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90 % · 8 Schritte**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · offiziell · 8 Schritte**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · offiziell · 8 Schritte**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 Schritte**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 Schritte**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 verwendet 80 % Videosparsity; Veda verwendet 90 %. Text und Audio behalten ihre ursprünglichen Regeln.[Parameter, SHA256 und Herkunft für alle zehn Videos](../examples/videos/qualitative-1344-345.json). Bei der vollständigen Dekodierung werden 345 Bilder und Audio überprüft; 16 einheitliche Bilder pro Clip wurden visuell überprüft. Es wird nicht behauptet, dass bestimmte Schüler pixel- oder wahrnehmungsmäßig gleichwertig sind. VSA behält Schussänderungen bei; Es wird keine Flimmerfreiheitsgarantie übernommen.

Die anfängliche Integration zielt auf Batch-One-T2VA ohne Referenzvideos ab. Siehe [Architektur](../docs/architecture.md),[Validierungsstatus](../docs/validation.md) und [Hinweise Dritter](../THIRD_PARTY_NOTICES.md).
