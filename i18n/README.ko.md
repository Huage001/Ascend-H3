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

[English](../README.md) | [简体中文](../README.zh-CN.md) | [繁體中文](../i18n/README.zh-TW.md) | **한국어** | [Español](../i18n/README.es.md) | [日本語](../i18n/README.ja.md) | [हिन्दी](../i18n/README.hi.md) | [मराठी](../i18n/README.mr.md) | [Русский](../i18n/README.ru.md) | [Português](../i18n/README.pt.md) | [తెలుగు](../i18n/README.te.md) | [Français](../i18n/README.fr.md) | [Deutsch](../i18n/README.de.md) | [Italiano](../i18n/README.it.md) | [Tiếng Việt](../i18n/README.vi.md) | [العربية](../i18n/README.ar.md) | [اردو](../i18n/README.ur.md) | [বাংলা](../i18n/README.bn.md) | [فارسی](../i18n/README.fa.md) | [Română](../i18n/README.ro.md) | [Türkçe](../i18n/README.tr.md)

</div>

![Ascend-H3 ecosystem](../docs/assets/ascend-h3-ecosystem.png)

> **License:** Project-authored contributions in this distribution are offered under the [PolyForm Noncommercial License 1.0.0](../LICENSE). Third-party code and model weights have separate terms, and earlier Apache-2.0 grants remain in effect. See [third-party notices](../THIRD_PARTY_NOTICES.md). For uses not covered by the applicable terms, contact [liusonghua@sjtu.edu.cn](mailto:liusonghua@sjtu.edu.cn).

> 이 완전한 README는 커밋 be5c42e03f9a에 있는 영어 README의 AI 지원 번역입니다. 구식이 될 수 있습니다. 최신 정보는 영어 버전을 참조하세요. [English](../README.md)

> **Translation notice:** Machine-translated and not technically reviewed; errors may occur. Use the English README as the authoritative reference for commands, requirements, and benchmark results.

하나의 저장소에서 전체, VSA, Veda, VDN 및 Spark-H3 관심을 통해 Ascend에 대한 MiniMax H3 추론. 통합에서는 vLLM-Omni 모델 로딩, INT8 변환기 양자화 및 기본 텐서 병렬 어텐션 헤드를 재사용합니다.

**검증된 범위:** 1344×768, 124/345 프레임 및 1/2/4/8 NPU의 40/40 INT8 E2E 셀(37개의 새로운 측정값과 3개의 명시적으로 표시된 기록 레이어 오프로드 셀). Ascend 910B3 및 PRO 6000의 상주 BF16 비교에는 20개의 어텐션 코어 측정과 20개의 전체 블록 측정이 포함되어 있습니다. 주의 연산자는 검증된 버전을 유지합니다.

## 기본 768p 워크로드

| 케이스 | 너비 × 높이 | 기본 출력 프레임 | 잠재 비디오 그리드 |
| --- | --- | --- | --- |
| `1344-124` | 1344×768 | 124 | 37×24×42 |
| `1344-345` | 1344×768 | 345 | 102×24×42 |

출력 FPS는 24입니다. 요청은 프레임 보간이 비활성화된 명시적 프레임 수를 사용합니다. 긴 사례는 합의된 기본 345프레임 워크로드입니다. 341프레임으로 잘리지 않습니다. BF16 및 INT8는 이 두 가지 1344×768 워크로드를 사용합니다. `--case`를 생략하면 `1344-124`가 선택됩니다.

<a id="install"></a>

## 설치

Linux, Python 3.12 및 호환 가능한 Ascend Torch/CANN 환경을 사용하세요.[환경 참고 사항](../docs/environment.md)에 설명된 고정된 vLLM, vLLM-Ascend 및 vLLM-Omni 스택을 설치합니다. 관련되지 않은 프레임워크 릴리스를 결합하지 마십시오. 그런 다음 소스에서 이 저장소를 설치합니다.

```bash
git clone https://github.com/Huage001/Ascend-H3.git
cd Ascend-H3
source /path/to/Ascend/cann/set_env.sh
bash scripts/install_stack.sh /path/to/private/framework-sources
export H3_ASCEND_CACHE=/path/to/writable/ascend-h3-cache
h3-ascend build --method all --soc Ascend910B3
```

빌드에는 모델 가중치가 없으며 NPU를 차지하지 않고 실행됩니다. 네이티브 라이브러리는 로컬로 컴파일되어 체크아웃 외부에 보관됩니다. 단일 백엔드의 경우 `all`를 `vsa`, `veda` 또는 `spark`로 바꿉니다. 전체 및 VDN에는 이러한 사용자 정의 CANN 확장이 필요하지 않습니다. 편집 가능한 체크아웃을 계속 사용 가능하게 유지하세요.

<a id="models"></a>

## 체크포인트 및 알고리즘 선택

| 방법 | 필수 모델 자산 | 계산 |
| --- | --- | --- |
| `full` | H3 베이스; LightX2V 기본 8단계 또는 커뮤니티 LoRA | 집중적인 관심 |
| `vsa` | 훈련된 압축 게이트 50개를 모두 포함하여 공식 FastH3 V2 학생을 완성하세요. | 8개 전달, 80% 비디오 희소성; 논리 KV64, 선택된 KV 블록이 칩의 KV128과 쌍을 이룹니다. |
| `veda` | H3 베이스 + LightX2V 8단계 + 공식 훈련된 Veda 예측기 | 동적 헤드당 Veda 블록 경로, 공칭 90% 희소성 |
| `vdn` | H3 베이스 + 공식 VDN 스테이지-DMD 체크포인트 디렉토리 | 로컬 소프트맥스와 128차원 학습된 DeltaNet 분기 |
| `spark` | H3 베이스 + LightX2V 8단 | Spark-H3 계층적 파티션 및 조인트 정확한/프록시 소프트맥스, 공칭 90% 정확한 분기 희소성 |

원본 라이선스에 따라 모델을 다운로드하세요. 가중치는 재분배되지 않습니다. 다양한 증류된 학생들은 서로 다른 일정과 학습된 규칙을 가지고 있습니다. 임의의 가중치에 주의를 집중한다고 해서 일치하는 품질이 보장되는 것은 아닙니다. VSA에는 0이 아닌 훈련된 게이트가 필요합니다. VDN에는 학습된 분기 가중치가 필요합니다. 다운로드 및 준비 명령은[모델 자산](../docs/weights.md)에 있습니다.

전체, Veda 및 Spark는 기본적으로 고정된 **LightX2V 8단계 LoRA**를 활성화합니다. 원래의 다단계 모델을 실행하려면 `--no-few-step-lora`를 사용하세요. 준비된 커뮤니티 `--lora /models/style`는 로드가 완료되기 전에 LightX2V와 함께 융합됩니다. 다른 커뮤니티 학생을 명시적으로 선택할 때 기본 학생을 비활성화합니다. Veda에는 항상 공식적으로 훈련된 예측자가 필요합니다. VSA는 ​​완전한 공식 **FastH3 V2 전체 체크포인트**: 8개 전달, 80% 비디오 희소성, 비디오/오디오 이동 10/3, 타일64, 안내 1만 허용합니다. 릴리스된 가중치를 직접 로드하고 `--lora`를 사용하지 않습니다. VDN는 완전한 공식 단계 DMD 체크포인트만 허용합니다. 추가 커뮤니티 LoRA 또는 학생 비활성화 플래그도 허용되지 않습니다.

```bash
# Optional: provide the exact pinned LightX2V source locally instead of downloading.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 \
  --few-step-source /models/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors
# Optional original 50-step model; the reporting baseline is Full eight-step.
h3-ascend serve --method full --model /models/MiniMax-H3 --devices 1 --no-few-step-lora
# General already-scaled Diffusers community LoRA, with no forced step schedule.
h3-ascend prepare-lora --source /models/style.safetensors --output /models/style
```

공식 PEFT 소스는 선언된 알파/순위를 사용하여 정규화됩니다. 준비된 파일은 해시 검사됩니다. 선택된 모든 어댑터는 TP 슬라이싱 및 INT8 양자화 전에 병합됩니다. 요청 시간 LoRA 행렬 곱셈은 없습니다. 소스 형식 및 고정 일정은[모델 준비](../docs/weights.md)를 참조하세요.

<a id="serve"></a>

## 서버 시작

다른 컴퓨팅 프로세스가 없는 장치만 사용하십시오. 카드 한 장의 경우:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 1 --offload text
```

다른 백엔드는 동일한 서버 API를 사용합니다.

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

선택한 장치에서 한 번에 하나의 서버를 실행합니다. 모델을 로드하지 않고 생성된 기본 vLLM-Omni 명령을 검사하려면 `--dry-run`를 추가하세요.

단일 카드 예제는 INT8 DiT를 상주하면서 텍스트 인코더를 스트리밍합니다. 기본 레시피는 DiT를 양자화합니다. 텍스트 인코더는 BF16로 유지되므로 해당 메모리 공간은 여전히 ​​시작 정책에 포함되어야 합니다. **BF16**의 경우 `--precision bf16`를 사용하세요. 필요한 경우 `--offload layer --no-precompute-adaln`를 명시적으로 선택하십시오. INT8는 기본 레시피를 사용합니다.

```json
{"transformer":{"method":"int8"}}
```

INT8 예측은 BF16 Q/K/V의 관심을 끌었습니다. Attention Accumulator, Veda 예측 및 VDN 상태는 규정된 정밀도를 유지합니다. 이는 INT8 모델 추론이지 모든 주의 작업이 INT8라는 주장은 아닙니다. 기본 레시피는 BF16에서 지원되지 않는 레이어를 유지합니다. 단일 카드 H3에는 NPU 양자화 매트뮬 출력 제한인 65,535를 초과하는 96,768 너비 프로젝션이 있습니다. TP는 순위당 너비를 범위로 줄일 수 있습니다.

**4카드 추론**의 경우 기본 텐서 병렬 헤드 샤딩을 사용합니다.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
h3-ascend serve --method spark --model /models/MiniMax-H3 \
  --precision int8 --devices 4 --parallel tp

# Eight cards: seven DiT attention heads per rank.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
h3-ascend serve --method full --model /models/MiniMax-H3 \
  --precision int8 --devices 8 --parallel tp
```

`--devices`는 56개의 헤드를 나누는 1, 2, 4, 7, 8을 지원합니다. Veda는 샤딩 후 글로벌 예측기 헤드 ID를 유지합니다. 세븐 카드 DiT TP는 64개의 쿼리/8 KV 헤드를 7로 나눌 수 없기 때문에 하나의 텍스트 인코더 순위를 사용합니다. 기본적으로 `--offload none`(상주 가중치)를 시작합니다. 메모리가 부족한 경우 `--offload text`는 장치에 DiT를 유지하면서 텍스트 인코더만 스트리밍합니다. `--offload module`는 전체 구성 요소를 교체합니다(실험 적격성). `--offload layer`는 ​​두 구성 요소를 모두 스트리밍하여 하나의 활성 블록과 하나의 프리패치된 블록을 유지합니다. 아래의 완성된 Matched-Offload 참조는 두 하드웨어 플랫폼 모두에서 `--offload layer`(`omni-layer-prefetch1-vae-resident-v1`)를 명시적으로 사용합니다. 새로운 엔드투엔드 측정에서는 **INT8만** 사용하며 상주 실행부터 시작하여 필요한 모든 오프로드 폴백을 표시합니다. 격리된 주의 및 DiT 블록 비교는 두 장치 모두에서 BF16를 사용합니다. 즉, 레이어 0/24/49, 각각 3번 실행되어 9개의 타이밍을 합산하고 9로 나눕니다. 선택적 `--parallel ulysses`는 VDN가 아닌 경로에 대해 노출됩니다. TP는 기본값으로 유지됩니다. 기본 타일 VAE 디코딩은 이제 선택한 모든 카드(`--vae-parallel-size 0`)에 기본값으로 적용됩니다. 단일 디코더 랭크에는 `--vae-parallel-size 1`를 사용하세요. INT8 전체/124프레임 파이프라인은 1, 2, 4, 8개의 VAE 랭크를 통과했습니다. 다른 방법/프레임 수 조합은 E2E 매트릭스에 남아 있습니다.

상주 NPU는 서버 준비 전에 AdaLN 결과를 준비합니다. 준비는 로드된 모델을 확인하고 LoRA의 실제 시그마 위치, DiT 업데이트 횟수 및 비디오/오디오 이동을 병합합니다. 8단계로 하드 코딩되지 않았습니다. `serve --steps N`를 사용하여 교사/사용자 지정 학생 시작을 지정하고 `generate --steps N`에서 동일한 개수를 사용합니다. 충돌하는 증류 체크포인트 일정은 거부됩니다. 필요한 계획에 따라 캐시 용량이 늘어납니다. 원래 투영 수학은 유지됩니다. AdaLN 마스터 가중치는 CPU에 유지되고 준비된 결과는 NPU에 유지됩니다. TP4/TP8은 또한 다운스트림 캐시를 무효화하는 낮은 비트 FP32 감소 변형을 방지하기 위해 정확한 시간 단계 입력에서 기본 시간 임베딩을 기억합니다. 가중치, 수치 설정 및 전체 순위 적중 합의는 여전히 재사용을 보호합니다. 다른 일정을 미리 계산하려면 새 구성으로 다시 시작하세요. 준비되지 않은 입력은 원래 계산을 사용하고 가중치를 전달할 수 있습니다. `--no-precompute-adaln`는 준비를 비활성화합니다. DiT 모듈/레이어 스트리밍은 기본적으로 비활성화되어 있습니다.

단일 카드 345프레임 Spark/VSA/VDN의 경우 원래 검증된 연산자와 함께 `--offload layer --no-precompute-adaln`를 사용하세요. E2E 테이블은 이러한 기본 계층 오프로드 셀을 †L로 표시합니다. ‡R은 이전 실행에서 재사용된 일치하는 측정값을 식별합니다. 다른 모든 셀은 개별적으로 기록된 거주 정책을 유지합니다.

런처는 기본 작업자 요청 기한을 7200초로 설정하므로 업스트림 기본값인 600초로 인해 긴 비디오가 중단되지 않습니다. 이 제한을 무시하려면 시작하기 전에 `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT`를 설정하십시오. 모델 계산이나 측정된 요청 기간이 아닌 대기 동작을 변경합니다.

## 두 개의 768p 케이스 생성

UTF-8 프롬프트 파일을 만들고 서버를 시작한 다음 일치하는 걸음 수를 사용합니다.

```bash
printf '%s\n' 'A cyclist rides through a green park, cinematic tracking shot.' > prompt.txt
for CASE in 1344-124 1344-345; do
  h3-ascend generate --case "$CASE" --prompt-file prompt.txt \
    --method full --seed 7 --output "outputs/${CASE}.mp4"
done
```

서버의 `--method`를 클라이언트에 전달합니다. 5가지 방법(Full/VSA/Veda/VDN/Spark)은 모두 기본적으로 8개의 실제 전달로 설정됩니다. 옵션인 원래 다단계 모델의 경우 서버와 일치하는 `generate --method full --no-few-step-lora`(50단계)를 사용하십시오. 별도로 준비된 커뮤니티 학생은 명시적인 `--steps`를 계속 사용할 수 있습니다. `--guidance-scale`는 선택 사항입니다. 생략하면 기본 정책이 유지됩니다. Refine 단계가 추가되지 않습니다. 클라이언트는 MP4와 JSON 타이밍 영수증을 저장합니다.

<!-- BEGIN BF16 LAYER RESULTS -->

## BF16 레이어 비교: 두 장치 모두에 상주

1344×768, 124/345 프레임만 해당. 실제 학생 가중치와 바이트가 동일한 고정 합성 은닉 상태 이는 비디오 요청에서 캡처된 레이어 활성화가 아닙니다. 레이어 0, 24, 49는 각각 2번의 시간 제한 없는 워밍업과 3번의 공식 실행을 받습니다. **9개의 타이밍을 합산하고 9로 나눕니다**. 가중치는 두 장치 모두에 그대로 유지됩니다. 활성화/주요 가중치는 BF16입니다. 필수 FP32 예측자, 누적 및 VDN 상태가 유지됩니다. 속도 향상은 동일한 하드웨어, 동일한 모양의 BF16 전체 기준을 사용합니다. 상대 속도 향상이 높을수록 대담합니다. 절대 대기 시간도 표시됩니다.

### 어텐션 코어만

준비된 Q/K/V로 시작하여 주의 출력으로 끝납니다. QKV 투영, Q/K 표준, RoPE, 출력 투영, AdaLN 및 FFN은 제외됩니다. 필요한 희소 게이트, 예측, 선택, 재배열 및 복원이 포함됩니다. VDN에는 완전한 소프트맥스 및 선형 코어가 모두 포함되어 있습니다. 모든 방법은 이 경계를 사용합니다. 전체 모듈 타이밍은 대체되지 않습니다. 코어 타이밍은 10/10 일치 하드웨어 쌍을 완료했습니다. 보류 중인 셀은 이전 모듈 측정을 재사용하지 않습니다.

| 프레임 | 방법 | Ascend 대기 시간 | PRO 6000 대기 시간 | Ascend 속도 향상 | PRO 6000 속도 향상 |
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

### DiT 블록 완성

예측, 정규화, RoPE, 주의 코어, AdaLN, FFN 및 잔여 작업을 포함하여 블록 입력에서 블록 출력까지 실행됩니다. 코어와 독립적으로 측정됩니다. 이들의 차이는 FFN 측정이 아니며 E2E 시간으로 추정되지도 않습니다.

| 프레임 | 방법 | Ascend 대기 시간 | PRO 6000 대기 시간 | Ascend 속도 향상 | PRO 6000 속도 향상 |
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

![BF16 주목 코어와 DiT 블록 비교](../docs/figures/bf16-three-layer.png)[모든 9회 실행 기록 및 입력 신원 확인](../benchmarks/bf16-three-layer.json).

<!-- END BF16 LAYER RESULTS -->

<!-- BEGIN INT8 E2E MATRIX -->

<a id="performance"></a>

## INT8 E2E: Ascend 1/2/4/8 카드

1344×768, 텍스트 인코딩, 비디오/오디오 디코딩 및 MP4 쓰기를 포함한 8개의 실제 DiT 업데이트. 로딩, 이전 LoRA 병합, AdaLN 준비 및 준비는 제외됩니다. 상주 DiT를 선호합니다. 상주 행은 공식 요청과 관련된 모든 순위에서 제로 AdaLN 누락 및 가중치 이동을 확인합니다. OOM을 기록한 후 표시된 단일 카드 345프레임 셀은 기본 레이어 오프로드를 사용하고 일치를 재사용하며 요청에 따라 완전히 검증된 이전 측정을 수행합니다. 이러한 셀은 새로운 측정값으로 표시되지 않고 명시적으로 표시됩니다.

| 프레임 | 방법 | 카드 1장 | 카드 2장 | 카드 4장 | 카드 8장 |
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

적용 **40/40**: 37개는 신선하고 3개는 재사용된 이전 측정값입니다. †T: 텍스트 인코더 계층 오프로드 전용; †L: 기본 DiT/텍스트 인코더 계층 1계층 프리패치 및 상주 VAE를 사용한 오프로드; ‡R: 2026-10-31의 원래 측정값과 일치합니다. 표시되지 않은 셀에는 런타임 오프로드가 없습니다. 상주 행만 AdaLN를 미리 계산합니다. 레이어 오프로드 행은 DiT 거주 또는 제로 AdaLN 전송 청구를 하지 않습니다. VAE는 매치 카드 수의 순위를 매깁니다. DiT 헤드는 56/28/14/7로 분할됩니다. 기본 INT8 적용 범위는 유지됩니다.[INT8 E2E 레코드](../benchmarks/ascend-int8-e2e.json)

<!-- END INT8 E2E MATRIX -->

이전에 일치된 레이어-오프로드 비교는[역사적 단일 카드 참조](../docs/archive/matched-offload-single-card.md)에 유지됩니다. 현재 테이블의 재사용된 셀은 변경되지 않은 원본 레코드에 연결됩니다.

## 정성적 결과 및 검증

Ascend 910B3만 해당: **1344×768, 345 기본 프레임, 24FPS(14.375초, 약 14.4초)**, 오디오 포함.

다섯 가지 방법 모두 8개의 전달을 사용합니다. 왼쪽: BF16. 오른쪽: 기본 W8A8 INT8. 모든 샘플은 하나의 카드, 동일한 프롬프트, 공통 시드 **7** 및 일치하는 오프로드를 사용합니다.

> 자전거 타는 사람은 조용한 녹색 공원 길, 부드러운 영화 추적 샷, 자연광을 따라 타고 있습니다.

<!-- BEGIN QUALITATIVE GALLERY -->
<table>
<tr><th width="50%">BF16</th><th width="50%">INT8 (W8A8)</th></tr>
<tr>
<td>

**Full + LightX2V8 · 8단계**

https://github.com/user-attachments/assets/1751d9fe-3ff9-48d1-a0dd-9ed62d431d17

</td>
<td>

**Full + LightX2V8 · 8단계**

https://github.com/user-attachments/assets/0514e9a0-76dd-4946-b4b8-b39ca38d7c69

</td>
</tr>
<tr>
<td>

**VSA · FastH3 V2 · 80% · 8단계**

https://github.com/user-attachments/assets/4a03fd28-4c88-4b46-b3aa-7dcb4ec4b1c1

</td>
<td>

**VSA · FastH3 V2 · 80% · 8단계**

https://github.com/user-attachments/assets/bae64455-0bbc-4708-b33e-1e1a4269e4cf

</td>
</tr>
<tr>
<td>

**Veda + LightX2V8 · 90% · 8단계**

https://github.com/user-attachments/assets/2c345f2b-4188-48d1-b0fc-6a44dd59800b

</td>
<td>

**Veda + LightX2V8 · 90% · 8단계**

https://github.com/user-attachments/assets/bc8e2eb5-b51b-4a53-bf9b-05c432dfe62a

</td>
</tr>
<tr>
<td>

**VDN · 공식 · 8단계**

https://github.com/user-attachments/assets/5e4d2ced-da14-4e50-b7ac-6ff1d298d478

</td>
<td>

**VDN · 공식 · 8단계**

https://github.com/user-attachments/assets/7069c2e4-844c-497b-a824-7a26a515dce2

</td>
</tr>
<tr>
<td>

**Spark-H3 + LightX2V8 · 8단계**

https://github.com/user-attachments/assets/66032d79-cf29-499e-ad0e-3f1bcaa753a9

</td>
<td>

**Spark-H3 + LightX2V8 · 8단계**

https://github.com/user-attachments/assets/c4a787a9-c1e9-46df-97f1-2887c7e49135

</td>
</tr>
</table>
<!-- END QUALITATIVE GALLERY -->

VSA V2는 80% 비디오 희소성을 사용합니다. Veda는 90%를 사용합니다. 텍스트와 오디오는 기본 규칙을 유지합니다.[10개 동영상 모두에 대한 매개변수, SHA256 및 출처](../examples/videos/qualitative-1344-345.json). 전체 디코딩은 345 프레임과 오디오를 확인합니다. 클립당 16개의 균일한 프레임을 시각적으로 검토했습니다. 서로 다른 학생들이 픽셀 또는 지각적으로 동등하다고 주장되지 않습니다. VSA는 샷 변경을 유지합니다. 깜박임 없음이 보장되지 않습니다.

초기 통합은 참조 비디오가 없는 배치 1 T2VA를 대상으로 합니다.[아키텍처](../docs/architecture.md),[검증 상태](../docs/validation.md) 및[타사 공지](../THIRD_PARTY_NOTICES.md)를 참조하세요.
