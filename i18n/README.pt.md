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

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | [한국어](../i18n/README.ko.md) | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | **Português** | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> Este README completo é uma tradução assistida por IA do README em inglês no commit be5c42e03f9a. Pode ficar desatualizado; consulte a versão em inglês para obter as informações mais recentes. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

Inferência MiniMax H3 em Ascend com atenção completa, VSA, Veda, VDN e Spark-H3 em um repositório. A integração reutiliza o carregamento do modelo vLLM-Omni, a quantização do transformador INT8 e os cabeçotes de atenção paralelos do tensor nativo.

**Escopo validado:** 40/40 células INT8 E2E em 1344×768, 124/345 quadros e 1/2/4/8 NPUs (37 medições recentes e 3 células históricas de descarregamento de camada explicitamente marcadas). A comparação BF16 residente em Ascend 910B3 e PRO 6000 contém 20 medições de núcleo de atenção e 20 medições de bloco completo. Os operadores de atenção mantêm suas versões validadas.

## Cargas de trabalho padrão de 768p

| Caso | Largura × altura | Quadros de saída nativos | Grade de vídeo latente |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

O FPS de saída é 24. As solicitações usam contagens de quadros explícitas, com a interpolação de quadros desabilitada. O caso mais longo é a carga de trabalho nativa acordada de 345 quadros; não é cortado para 341 quadros. BF16 e INT8 usam essas duas cargas de trabalho de 1344×768. A omissão de `--case` seleciona `1344-124`.

<a id="install"></a>

## Instalação

Use Linux, Python 3.12 e um ambiente Ascend Torch/CANN compatível. Instale a pilha fixada vLLM, vLLM-Ascend e vLLM-Omni descrita nas [notas do ambiente](../docs/environment.md). Não combine versões de estrutura não relacionadas. Em seguida, instale este repositório a partir do código-fonte:

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

As compilações não contêm pesos de modelo e são executadas sem ocupar uma NPU. Bibliotecas nativas são compiladas localmente e mantidas fora do checkout. Para um único back-end, substitua `all` por `vsa`, `veda` ou `spark`; Full e VDN não requerem essas extensões CANN personalizadas. Mantenha o checkout editável disponível.

<a id="models"></a>

## Pontos de verificação e escolhas de algoritmo

| Método | Ativos de modelo necessários | Computação |
| --- | --- | --- |
| `full` | base H3; LightX2V 8 etapas por padrão ou uma comunidade LoRA | Atenção densa |
| `vsa` | Aluno oficial completo do FastH3 V2, incluindo todos os 50 portões de compressão treinados | 8 encaminhamentos, 80% de escassez de vídeo; KV64 lógico, blocos KV selecionados emparelhados em KV128 no chip |
| `veda` | Base H3 + LightX2V 8 etapas + preditor Veda oficial treinado | Rotas de bloco Veda dinâmicas por cabeça, dispersão nominal de 90% |
| `vdn` | Base H3 + diretório oficial de ponto de verificação stage-DMD VDN | Softmax local mais ramificação DeltaNet aprendida de 128 dimensões |
| `spark` | Base H3 + LightX2V 8 etapas | Partição hierárquica Spark-H3 e articulação exata/proxy softmax, esparsidade nominal de 90% de ramificação exata |

Baixe modelos sob suas licenças originais. Os pesos não são redistribuídos. Diferentes alunos destilados têm horários e regras aprendidas diferentes; trocar a atenção em pesos arbitrários não garante a qualidade correspondente. VSA requer portas treinadas diferentes de zero. VDN requer seus pesos de ramificação aprendidos. Os comandos de download e preparação estão em [ativos de modelo](../docs/weights.md).

Full, Veda e Spark habilitam o **LightX2V LoRA** fixado de oito etapas por padrão. Use `--no-few-step-lora` para executar o modelo original de muitas etapas. Uma comunidade preparada `--lora /models/style` é fundida com LightX2V antes do carregamento terminar; desativar o aluno padrão ao selecionar explicitamente outro aluno da comunidade. Veda sempre precisa de seu preditor oficial treinado. VSA aceita apenas o ponto de verificação completo **FastH3 V2 completo**: oito para a frente, 80% de esparsidade de vídeo, mudanças de vídeo/áudio 10/3, tile64, orientação 1. Ele carrega os pesos liberados diretamente e não leva `--lora`. VDN aceita apenas seu ponto de verificação DMD oficial completo. Nenhum deles aceita uma comunidade adicional LoRA ou sinalizador de desativação de aluno.

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

A fonte PEFT oficial é normalizada uma vez usando seu alfa/classificação declarado; os arquivos preparados são verificados por hash. Todos os adaptadores selecionados são mesclados antes do fatiamento TP e da quantização INT8. Não há multiplicações de matriz LoRA no momento da solicitação. Consulte [preparação do modelo](../docs/weights.md) para formatos de origem e programações fixas.

<a id="serve"></a>

## Iniciar um servidor

Use apenas dispositivos que não tenham outros processos de computação. Para um cartão:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

Outros back-ends usam a mesma API do servidor:

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

Execute um servidor por vez nos dispositivos selecionados. Adicione `--dry-run` para inspecionar o comando vLLM-Omni nativo gerado sem carregar um modelo.

Os exemplos de cartão único transmitem o codificador de texto enquanto mantêm o INT8 DiT residente. A receita nativa quantiza o DiT; o codificador de texto permanece BF16, portanto, seu consumo de memória ainda precisa ser incluído na política de lançamento. Para **BF16**, use `--precision bf16`; se necessário, selecione `--offload layer --no-precompute-adaln` explicitamente. INT8 usa a receita nativa:

```json
{"transformer":{"method":"int8"}}
```

As projeções INT8 chamam a atenção de BF16 Q/K/V. Acumuladores de atenção, previsões Veda e estado VDN mantêm sua precisão prescrita. Esta é uma inferência do modelo INT8, não uma afirmação de que toda operação de atenção é INT8. A receita nativa retém camadas não suportadas em BF16. O cartão único H3 tem uma projeção de 96.768 larguras que excede o limite de saída matmul quantizado NPU de 65.535; O TP pode reduzir sua largura por classificação dentro do alcance.

Para **inferência de quatro cartas**, use fragmentação de cabeça paralela de tensor nativo:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices` suporta 1, 2, 4, 7 e 8, que dividem 56 cabeças. Veda retém IDs de cabeça do preditor global após a fragmentação. O DiT TP de sete placas usa uma classificação de codificador de texto porque seus cabeçotes de consulta 64/8 KV não podem ser divididos por sete. Inicia o padrão para `--offload none` (pesos residentes). Se a memória for insuficiente, `--offload text` transmite apenas o codificador de texto, mantendo o DiT no dispositivo; `--offload module` troca componentes completos (qualificação experimental); `--offload layer` transmite ambos os componentes, mantendo um bloco ativo e um bloco pré-buscado. A referência de descarregamento correspondente completa abaixo usa explicitamente `--offload layer` (`omni-layer-prefetch1-vae-resident-v1`) em ambas as plataformas de hardware. Novas medições ponta a ponta usam **somente INT8**, começando com a execução residente e marcando cada fallback de descarregamento necessário. A atenção isolada e a comparação do bloco DiT usam BF16 em ambos os dispositivos: camadas 0/24/49, três execuções cada, somando os nove tempos e dividindo por nove. `--parallel ulysses` opcional é exposto para caminhos não VDN; TP continua sendo o padrão. A decodificação VAE em mosaico nativo agora é padronizada para todos os cartões selecionados (`--vae-parallel-size 0`); use `--vae-parallel-size 1` para uma única classificação de decodificador. O pipeline INT8 completo/124 quadros passou com uma, duas, quatro e oito classificações VAE. Outras combinações de método/contagem de quadros permanecem na matriz E2E.

Os lançamentos de NPU residentes preparam os resultados AdaLN antes da preparação do servidor. A preparação resolve o modelo carregado e mescla as posições sigma reais de LoRA, contagem de atualização de DiT e mudanças de vídeo/áudio; não é codificado em oito etapas. Use `serve --steps N` para especificar um lançamento de professor/aluno personalizado e use a mesma contagem em `generate --steps N`. Uma programação de ponto de verificação destilada conflitante é rejeitada. A capacidade do cache aumenta com os planos necessários. A matemática da projeção original é mantida; Os pesos mestres AdaLN permanecem na CPU, enquanto os resultados preparados permanecem na NPU. TP4/TP8 também memoriza incorporações de tempo nativo a partir de entradas exatas de timestep para evitar variação de redução de FP32 de baixo bit, invalidando caches downstream. Pesos, configurações numéricas e acordo de acerto em todas as classificações ainda protegem a reutilização. Reinicie com a nova configuração para pré-calcular um agendamento diferente; uma entrada não preparada usa o cálculo original e pode transferir pesos. `--no-precompute-adaln` desativa a preparação; O streaming de módulo/camada DiT o mantém desabilitado por padrão.

Para Spark/VSA/VDN de placa única de 345 quadros, use `--offload layer --no-precompute-adaln` com os operadores validados originais. A tabela E2E marca essas células de descarregamento da camada nativa com †L; ‡R identifica medições correspondentes reutilizadas da execução anterior. Todas as outras células mantêm a sua política de residência registada individualmente.

O iniciador define o prazo de solicitação do trabalhador nativo para 7200 segundos, para que um vídeo longo não seja abortado pelo padrão upstream de 600 segundos. Defina `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` antes do lançamento para substituir esse limite; ele altera o comportamento de espera, não o cálculo do modelo ou a duração medida da solicitação.

## Gere os dois casos 768p

Crie um arquivo de prompt UTF-8, inicie um servidor e use a contagem de etapas correspondente:

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

Passe o `--method` do servidor para o cliente: todos os cinco métodos (Full/VSA/Veda/VDN/Spark) são padronizados para oito encaminhamentos reais. Para o modelo original opcional de muitas etapas, use `generate --method full --no-few-step-lora` (50 etapas), correspondente ao servidor. Um `--steps` explícito permanece disponível para um aluno da comunidade preparado separadamente. `--guidance-scale` é opcional; a omissão preserva a política nativa. Nenhum estágio Refinar é adicionado. O cliente salva o MP4 e um recibo de temporização JSON.

<!-- BEGIN BF16 LAYER RESULTS -->

## Comparação da camada BF16: residente em ambos os dispositivos

Somente quadros 1344×768, 124/345. Pesos reais dos alunos e estados ocultos sintéticos congelados com bytes idênticos; estas não são ativações de camada capturadas de uma solicitação de vídeo. As camadas 0, 24 e 49 recebem, cada uma, dois aquecimentos de duração indeterminada e três execuções formais: **soma os nove tempos e divide por nove**. Os pesos permanecem residentes em ambos os dispositivos. As ativações/pesos principais são BF16; os preditores FP32, acumulações e estado VDN necessários são retidos. Os aceleradores usam o mesmo hardware e mesmo formato BF16 Linha de base completa. A maior aceleração relativa é ousada; a latência absoluta também é mostrada.

### Atenção apenas núcleo

Começa com Q/K/V preparado e termina na saída de atenção. Exclui projeção QKV, norma Q/K, RoPE, projeção de saída, AdaLN e FFN. Inclui portas esparsas necessárias, previsão, seleção, reorganização e restauração; VDN inclui núcleos softmax e lineares completos. Todo método usa esse limite; as temporizações de todo o módulo não são substituídas. O tempo do núcleo completou 10/10 pares de hardware correspondentes; células pendentes não reutilizam medições anteriores do módulo.

| Molduras | Método | Latência Ascend | Latência PRO 6000 | Aceleração Ascend | Aceleração PRO 6000 |
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

### Bloco DiT completo

Funciona da entrada ao bloco de saída, incluindo projeções, normalização, RoPE, núcleo de atenção, AdaLN, FFN e operações residuais. Medido independentemente do núcleo; sua diferença não é uma medida FFN e nem é extrapolada para o tempo E2E.

| Molduras | Método | Latência Ascend | Latência PRO 6000 | Aceleração Ascend | Aceleração PRO 6000 |
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

![Comparação do núcleo de atenção BF16 e bloco DiT](../docs/figures/bf16-three-layer.png)[Todos os registros de nove execuções e verificações de identidade de entrada](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1/2/4/8 cartões

1344×768, oito atualizações reais do DiT, incluindo codificação de texto, decodificação de vídeo/áudio e gravação de MP4. Carregamento, mesclagem anterior de LoRA, preparação e aquecimento de AdaLN são excluídos. Prefira o residente DiT; as linhas residentes verificam zero falhas AdaLN e transferências de peso em cada classificação em torno da solicitação formal. Após OOMs registrados, as células de 345 quadros de cartão único marcadas usam correspondência de descarregamento e reutilização de camada nativa, medições anteriores totalmente validadas conforme solicitado; essas células são marcadas explicitamente em vez de apresentadas como medidas recentes.

| Molduras | Método | 1 cartão | 2 cartas | 4 cartas | 8 cartas |
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

Coberto **40/40**: 37 medições anteriores frescas e 3 reutilizadas. †T: apenas descarregamento da camada do codificador de texto; †L: DiT nativo/descarregamento da camada do codificador de texto com pré-busca de uma camada e VAE residente; ‡R: correspondendo às medidas originais de 2026-10-31. Células não marcadas não têm descarregamento de tempo de execução. Somente linhas residentes pré-calculam AdaLN; As linhas de descarregamento de camada não fazem nenhuma reivindicação de residência DiT ou de transferência zero de AdaLN. VAE classifica a contagem de cartas correspondentes; Cabeças DiT divididas em 56/28/14/7. A cobertura nativa INT8 é mantida.[Registros INT8 E2E](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

A comparação de descarregamento de camada correspondente anterior é mantida na [referência histórica de cartão único](../docs/archive/matched-offload-single-card.md). Células reutilizadas na tabela atual vinculam-se a registros originais inalterados.

## Resultados qualitativos e validação

Somente Ascend 910B3: **1344×768, 345 quadros nativos, 24 FPS (14,375 segundos, aproximadamente 14,4 segundos)**, com áudio.

Todos os cinco métodos usam oito atacantes. Esquerda: BF16. À direita: W8A8 nativo INT8. Todas as amostras usam um cartão, o mesmo prompt, seed comum **7** e descarregamento correspondente.

> Um ciclista anda por um caminho tranquilo de um parque verde, filmagem suave e luz natural.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Completo + LightX2V8 · 8 etapas**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Completo + LightX2V8 · 8 etapas**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8 etapas**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8 etapas**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8 etapas**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8 etapas**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · oficial · 8 etapas**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · oficial · 8 etapas**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8 etapas**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8 etapas**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2 usa 80% de dispersão de vídeo; Veda usa 90%. Texto e áudio mantêm suas regras nativas.[Parâmetros, SHA256 e procedência para todos os dez vídeos](../examples/videos/qualitative-1344-345.json). A decodificação completa verifica 345 quadros e áudio; 16 quadros uniformes por clipe foram revisados ​​visualmente. Alunos distintos não são considerados equivalentes em pixels ou perceptualmente. VSA retém alterações de tiro; nenhuma garantia de ausência de cintilação é feita.

A integração inicial visa o lote um T2VA sem vídeos de referência. Consulte [arquitetura](../docs/architecture.md),[status de validação](../docs/validation.md) e [avisos de terceiros](../THIRD_PARTY_NOTICES.md).
