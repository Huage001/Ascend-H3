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

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | **Español** | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> **License:** Project-authored contributions in this distribution are offered under the [PolyForm Noncommercial License 1.0.0](../LICENSE). Third-party code and model weights have separate terms, and earlier Apache-2.0 grants remain in effect. See [third-party notices](../THIRD_PARTY_NOTICES.md). For uses not covered by the applicable terms, contact [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn).

> Este README completo es una traducción asistida por IA del README en inglés en el compromiso be5c42e03f9a. Puede quedar obsoleto; consulte la versión en inglés para obtener la información más reciente. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

Inferencia MiniMax H3 en Ascend con atención completa, VSA, Veda, VDN y Spark-H3 en un repositorio. La integración reutiliza la carga del modelo vLLM-Omni, la cuantificación del transformador INT8 y los cabezales de atención paralelos tensoriales nativos.

**Alcance validado:** 40/40 celdas INT8 E2E a 1344×768, 124/345 fotogramas y 1/2/4/8 NPU (37 mediciones nuevas y 3 celdas de descarga de capa histórica marcadas explícitamente). La comparación residente BF16 en Ascend 910B3 y PRO 6000 contiene 20 mediciones de núcleo de atención y 20 de bloque completo. Los operadores de atención conservan sus versiones validadas.

## Cargas de trabajo predeterminadas de 768p

| caso | Ancho × alto | Marcos de salida nativos | Cuadrícula de vídeo latente |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

El FPS de salida es 24. Las solicitudes utilizan recuentos de fotogramas explícitos, con la interpolación de fotogramas desactivada. El caso largo es la carga de trabajo nativa acordada de 345 marcos; no está recortado a 341 fotogramas. BF16 y INT8 utilizan estas dos cargas de trabajo de 1344×768. Al omitir `--case` se selecciona `1344-124`.

<a id="install"></a>

## Instalación

Utilice Linux, Python 3.12 y un entorno Ascend Torch/CANN compatible. Instale la pila anclada vLLM, vLLM-Ascend y vLLM-Omni descrita en [las notas de entorno](../docs/environment.md). No combine versiones de marcos no relacionados. Luego instale este repositorio desde la fuente:

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

Las compilaciones no contienen pesos de modelo y se ejecutan sin ocupar una NPU. Las bibliotecas nativas se compilan localmente y se mantienen fuera del repositorio clonado. Para un único backend, reemplace `all` con `vsa`, `veda` o `spark`; Full y VDN no requieren estas extensiones CANN personalizadas. Mantenga disponible la copia editable del repositorio.

<a id="models"></a>

## Puntos de control y opciones de algoritmos

| Método | Activos de modelo requeridos | Computación |
| --- | --- | --- |
| `full` | base H3; LightX2V 8 pasos por defecto, o una comunidad LoRA | Atención densa |
| `vsa` | Estudiante oficial completo de FastH3 V2, incluidas las 50 puertas de compresión entrenadas | 8 delanteros, 80% de escasez de vídeo; KV64 lógico, bloques KV seleccionados emparejados en KV128 en el chip |
| `veda` | Base H3 + LightX2V de 8 pasos + predictor Veda entrenado oficialmente | Rutas de bloques dinámicas por cabeza Veda, escasez nominal del 90 % |
| `vdn` | Base H3 + directorio oficial de puntos de control VDN stage-DMD | Softmax local más rama DeltaNet aprendida de 128 dimensiones |
| `spark` | Base H3 + LightX2V 8 pasos | Spark-H3 partición jerárquica y softmax conjunto exacto/proxy, escasez nominal de rama exacta del 90 % |

Descarga modelos bajo sus licencias originales. Los pesos no se redistribuyen. Diferentes estudiantes destilados tienen diferentes horarios y reglas aprendidas; Intercambiar atención sobre pesos arbitrarios no garantiza una calidad coincidente. VSA requiere puertas entrenadas distintas de cero. VDN requiere los pesos de rama aprendidos. Los comandos de descarga y preparación se encuentran en [activos del modelo](../docs/weights.md).

Full, Veda y Spark habilitan de forma predeterminada el **LoRA LightX2V de ocho pasos** fijado. Utilice `--no-few-step-lora` para ejecutar el modelo original de muchos pasos. Una comunidad preparada `--lora /models/style` se fusiona con LightX2V antes de que finalice la carga; deshabilite el estudiante predeterminado al seleccionar explícitamente otro estudiante de la comunidad. Veda siempre necesita su predictor oficial capacitado. VSA acepta solo el punto de control completo oficial **FastH3 V2**: ocho hacia adelante, 80 % de dispersión de video, cambios de video/audio 10/3, mosaico 64, guía 1. Carga los pesos liberados directamente y no toma `--lora`. VDN acepta solo su punto de control DMD de etapa oficial completo. Ninguno acepta una comunidad adicional LoRA o una bandera de inhabilitación para estudiantes.

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

La fuente oficial PEFT se normaliza una vez utilizando su alfa/rango declarado; los archivos preparados se verifican mediante hash. Todos los adaptadores seleccionados se fusionan antes del corte TP y la cuantificación INT8. No hay multiplicaciones de matrices LoRA en tiempo de solicitud. Consulte [preparación del modelo](../docs/weights.md) para conocer los formatos de origen y los horarios fijos.

<a id="serve"></a>

## Iniciar un servidor

Utilice únicamente dispositivos que no tengan otros procesos informáticos. Para una tarjeta:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

Otros backends usan la misma API de servidor:

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

Ejecute un servidor a la vez en los dispositivos seleccionados. Agregue `--dry-run` para inspeccionar el comando vLLM-Omni nativo generado sin cargar un modelo.

Los ejemplos de una sola tarjeta transmiten el codificador de texto mientras mantienen residente el INT8 DiT. La receta nativa cuantifica el DiT; el codificador de texto sigue siendo BF16, por lo que su uso de memoria aún debe incluirse en la política de lanzamiento. Para **BF16**, utilice `--precision bf16`; si es necesario, seleccione `--offload layer --no-precompute-adaln` explícitamente. INT8 utiliza la receta nativa:

```json
{"transformer":{"method":"int8"}}
```

Las proyecciones INT8 ponen en atención BF16 Q/K/V. Los acumuladores de atención, las predicciones Veda y el estado VDN mantienen la precisión prescrita. Esta es una inferencia del modelo INT8, no una afirmación de que cada operación de atención sea INT8. La receta nativa conserva capas no compatibles en BF16. La tarjeta única H3 tiene una proyección de 96,768 de ancho que excede el límite de salida matmul cuantificado de NPU de 65,535; TP puede reducir su ancho por rango al rango.

Para **inferencia de cuatro cartas**, utilice la fragmentación de cabezas paralelas del tensor nativo:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices` admite 1, 2, 4, 7 y 8, que dividen 56 cabezas. Veda conserva los ID de cabeza de predictor global después de la fragmentación. El TP DiT de siete tarjetas utiliza un rango de codificador de texto porque sus 64 cabezales de consulta/8 KV no se pueden dividir por siete. Inicia de forma predeterminada `--offload none` (pesos residentes). Si la memoria es insuficiente, `--offload text` transmite solo el codificador de texto mientras conserva el DiT en el dispositivo; `--offload module` intercambia componentes completos (calificación experimental); `--offload layer` transmite ambos componentes, manteniendo uno activo y otro precargado. La referencia de descarga coincidente completa que aparece a continuación utiliza explícitamente `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) en ambas plataformas de hardware. Las nuevas mediciones de un extremo a otro utilizan **INT8 únicamente**, comenzando con la ejecución residente y marcando cada respaldo de descarga necesario. La atención aislada y la comparación de bloques DiT utilizan BF16 en ambos dispositivos: capas 0/24/49, tres ejecuciones cada una, sumando los nueve tiempos y dividiéndolos por nueve. El `--parallel ulysses` opcional está expuesto para rutas que no son VDN; TP sigue siendo el predeterminado. La decodificación nativa en mosaico VAE ahora está predeterminada en todas las tarjetas seleccionadas (`--vae-parallel-size 0`); utilice `--vae-parallel-size 1` para un único rango de decodificador. El pipeline INT8 Full/124-frame ha pasado con uno, dos, cuatro y ocho rangos VAE. Otras combinaciones de método/recuento de fotogramas permanecen en la matriz E2E.

Los lanzamientos de NPU residentes preparan los resultados de AdaLN antes de que el servidor esté listo. La preparación resuelve el modelo cargado y fusiona las posiciones sigma reales de LoRA, el recuento de actualizaciones de DiT y los cambios de video/audio; no está codificado en ocho pasos. Utilice `serve --steps N` para especificar un inicio de profesor/estudiante personalizado y utilice el mismo recuento en `generate --steps N`. Se rechaza un cronograma de puntos de control destilado contradictorio. La capacidad de caché crece con los planes necesarios. Se conservan las matemáticas de proyección originales; Los pesos maestros AdaLN permanecen en la CPU, mientras que los resultados preparados permanecen en la NPU. TP4/TP8 también memoriza incrustaciones de tiempo nativo a partir de entradas de pasos de tiempo exactos para evitar que la variación de reducción de FP32 de bits bajos invalide los cachés posteriores. Los pesos, las configuraciones numéricas y el acuerdo de impacto de todos los rangos aún protegen la reutilización. Reinicie con la nueva configuración para precalcular un cronograma diferente; una entrada no preparada utiliza el cálculo original y puede transferir pesos. `--no-precompute-adaln` desactiva la preparación; La transmisión de módulo/capa DiT lo mantiene desactivado de forma predeterminada.

Para Spark/VSA/VDN de 345 cuadros de tarjeta única, utilice `--offload layer --no-precompute-adaln` con los operadores validados originales. La tabla E2E marca estas celdas de descarga de capa nativa con †L; ‡R identifica mediciones coincidentes reutilizadas de la ejecución anterior. Todas las demás células conservan su política de residencia registrada individualmente.

El iniciador establece la fecha límite de solicitud de trabajador nativo en 7200 segundos para que un video largo no sea cancelado por el valor predeterminado de 600 segundos ascendente. Configure `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` antes del lanzamiento para anular este límite; cambia el comportamiento de espera, no el cálculo del modelo o la duración medida de la solicitud.

## Genera los dos casos de 768p.

Cree un archivo de solicitud UTF-8, inicie un servidor y luego use su recuento de pasos correspondiente:

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

Pase el `--method` del servidor al cliente: los cinco métodos (Full/VSA/Veda/VDN/Spark) tienen de forma predeterminada ocho reenvíos reales. Para el modelo original opcional de muchos pasos, utilice `generate --method full --no-few-step-lora` (50 pasos), que coincide con el servidor. Un `--steps` explícito permanece disponible para un estudiante de la comunidad preparado por separado. `--guidance-scale` es opcional; la omisión preserva la política nativa. No se agrega ninguna etapa de Refinamiento. El cliente guarda el MP4 y un recibo de sincronización JSON.

<!-- BEGIN BF16 LAYER RESULTS -->

## Comparación de capas BF16: residente en ambos dispositivos

1344×768, 124/345 fotogramas únicamente. Pesos reales de estudiantes y estados ocultos sintéticos congelados de bytes idénticos; Estas no son activaciones de capas capturadas a partir de una solicitud de video. Las capas 0, 24 y 49 reciben cada una dos calentamientos no cronometrados y tres carreras formales: **sume los nueve tiempos y divídelos por nueve**. Los pesos permanecen residentes en ambos dispositivos. Las activaciones/pesos principales son BF16; Se conservan los predictores FP32 requeridos, las acumulaciones y el estado VDN. Las aceleraciones utilizan el mismo hardware y la misma forma BF16 Línea base completa. La mayor aceleración relativa está en negrita; También se muestra la latencia absoluta.

### Solo núcleo de atención

Comienza con Q/K/V preparado y termina en la salida de atención. Excluye proyección QKV, norma Q/K, RoPE, proyección de salida, AdaLN y FFN. Incluye puertas dispersas requeridas, predicción, selección, reordenamiento y restauración; VDN incluye núcleos softmax y lineales completos. Cada método utiliza este límite; los tiempos de todo el módulo no se sustituyen. La sincronización central ha completado 10/10 pares de hardware coincidentes; Las celdas pendientes no reutilizan mediciones anteriores del módulo.

| Marcos | Método | Latencia Ascend | Latencia PRO 6000 | Aceleración Ascend | Aceleración PRO 6000 |
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

### Bloque completo DiT

Se ejecuta desde la entrada del bloque hasta la salida del bloque, incluidas proyecciones, normalización, RoPE, núcleo de atención, AdaLN, FFN y operaciones residuales. Medido independientemente del núcleo; su diferencia no es una medida FFN y tampoco se extrapola al tiempo E2E.

| Marcos | Método | Latencia Ascend | Latencia PRO 6000 | Aceleración Ascend | Aceleración PRO 6000 |
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

![BF16 núcleo de atención y comparación de bloques DiT](../docs/figures/bf16-three-layer.png)[Todos los registros de nueve ejecuciones y comprobaciones de identidad de entrada](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1 / 2 / 4 / 8 tarjetas

1344×768, ocho actualizaciones reales de DiT, incluida codificación de texto, decodificación de vídeo/audio y escritura MP4. Se excluyen la carga, la fusión previa de LoRA, la preparación de AdaLN y los calentamientos. Prefiere residente DiT; Las filas residentes verifican que no haya errores AdaLN ni transferencias de peso en cada fila alrededor de la solicitud formal. Después de los OOM registrados, las celdas de 345 fotogramas de una sola tarjeta marcadas utilizan la descarga de la capa nativa y la reutilización, mediciones previas completamente validadas según lo solicitado; estas celdas están marcadas explícitamente en lugar de presentarse como mediciones nuevas.

| Marcos | Método | 1 tarjeta | 2 cartas | 4 cartas | 8 cartas |
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

Cubierto **40/40**: 37 mediciones anteriores frescas y 3 reutilizadas. †T: capa de codificador de texto Solo descarga; †L: descarga nativa de capa DiT/codificador de texto con captación previa de una capa y VAE residente; ‡R: coincide con las medidas originales de 2026-10-31. Las celdas sin marcar no tienen descarga en tiempo de ejecución. Sólo las filas residentes precalculan AdaLN; Las filas de descarga de capa no hacen ningún reclamo de residencia DiT o transferencia cero-AdaLN. VAE clasifica el recuento de cartas del partido; Cabezales DiT divididos 56/28/14/7. Se conserva la cobertura nativa INT8.[INT8 registros E2E](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

La comparación anterior de descarga de capas coincidentes se conserva en la [referencia histórica de una sola tarjeta](../docs/archive/matched-offload-single-card.md). Las celdas reutilizadas en la tabla actual se vinculan a registros originales sin cambios.

## Resultados cualitativos y validación.

Solo Ascend 910B3: **1344×768, 345 fotogramas nativos, 24 FPS (14,375 segundos, aproximadamente 14,4 segundos)**, con audio.

Los cinco métodos utilizan ocho delanteros. Izquierda: BF16. Derecha: W8A8 INT8 nativo. Todas las muestras utilizan una tarjeta, el mismo mensaje, semilla común **7** y descarga coincidente.

> Un ciclista recorre un tranquilo sendero verde del parque, un suave seguimiento cinematográfico y luz natural.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Completo + LightX2V8 · 8 pasos**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Completo + LightX2V8 · 8 pasos**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 pasos**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 pasos**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 pasos**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 pasos**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · oficial · 8 pasos**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · oficial · 8 pasos**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 pasos**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 pasos**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 utiliza un 80 % de escasez de vídeo; Veda utiliza el 90%. El texto y el audio conservan sus reglas nativas.[Parámetros, SHA256 y procedencia de los diez vídeos](../examples/videos/qualitative-1344-345.json). La decodificación completa verifica 345 fotogramas y audio; Se revisaron visualmente 16 fotogramas uniformes por clip. No se afirma que los distintos estudiantes sean equivalentes en términos de píxeles o percepción. VSA conserva los cambios de disparo; no se ofrece ninguna garantía sin parpadeos.

La integración inicial apunta al lote uno de T2VA sin videos de referencia. Consulte [arquitectura](../docs/architecture.md),[estado de validación](../docs/validation.md) y [avisos de terceros](../THIRD_PARTY_NOTICES.md).
