# Moshi Custom Layers: PyTorch → Flax/JAX for TPU

이 저장소는 **`modeling_moshi.py`**의 핵심 커스텀 레이어들을 **TPU 최적화**를 위해 **PyTorch에서 Flax/JAX로 변환**하고, 두 버전이 **동일하게 동작**하는지 확인하기 위한 단위 테스트를 담고 있습니다.

## 주요 배경

- TPU 환경에서 **JAX/Flax**가 PyTorch 대비 높은 효율을 내는 경우가 많아, **커스텀 연산 레이어**를 우선 변환 후 검증 중입니다.  
- 우선 변환 대상 커스텀 레이어:
  1. **MoshiFlexibleLinear**  
  2. **MoshiRMSNorm**  
  3. **MoshiGatingMLP**  
  4. **MoshiRotaryEmbedding** (RoPE)  
  5. **MoshiAttention**  

(위 레이어들은 Hugging Face [transformers/models/moshi/modeling_moshi.py]에서 가져온 커스텀 구현체 기반)

## 현재 구성

1. **`class *PT`** : PyTorch 구현체  
2. **`class *FL`** : Flax/JAX 구현체  
3. **가중치 복사 & 결과 비교**:  
   - PyTorch 레이어의 weight를 Flax로 로드 → **forward** 출력 비교(평균 차이 등)

다양한 테스트 스크립트를 통해, 변환된 레이어/모델들이 PyTorch와 Flax에서 같은 로직과 가중치로 동일(혹은 매우 근접)한 출력을 내는지 검증합니다.

### 주요 테스트 스크립트

- **`layer_unit_test_TPU_compare3.py`** 등 :  
  - FlexibleLinear, RMSNorm, GatingMLP 등 개별 레이어 단위 기본 테스트  
- **`layer_unit_test_TPU_att_rope_compare3.py`** :  
  - MoshiAttention, RoPE(회전 위치 인코딩) 관련 PyTorch↔Flax 비교  
- **`layer_unit_test_MoshiDecoderLayer.py`** :  
  - **`MoshiDecoderLayerPT`** vs **`MoshiDecoderLayerFL`** 구현 확인  
  - RMSNorm, GatingMLP, Self-Attention, RoPE 로직을 조합한 **디코더 레이어** 단위의 PyTorch↔Flax 매핑 테스트  
  - Flexible Linear 및 레이어 인덱스(layer_idx) 등을 통해 코드북 사용 시나리오 검증  
- **`layer_unit_test_MoshiModel.py`** :  
  - **`MoshiModelPT`** vs **`MoshiModelFL`** (임베딩 + 다중 디코더 레이어 + 최종 Norm) 구조 확인  
  - Flexible Linear, Codebook 개수 설정 등 다양한 경우(토큰별 다른 codebook 인덱스 포함)에서 PyTorch↔Flax의 출력 정밀도 확인

## 사용 방법

1. **Python 환경 준비**  
   - PyTorch, JAX, Flax, NumPy 설치  
   - TPU 환경이라면 TPU용 jaxlib 준비  

2. **테스트 스크립트 실행**
   ```bash
   # 기본 레이어들(FlexibleLinear, RMSNorm, GatingMLP 등)
   python layer_unit_test_TPU_compare3.py
   python layer_unit_test_TPU_att_rope_compare3.py
   
   # 디코더 레이어 전체 테스트
   python layer_unit_test_MoshiDecoderLayer.py

   # 모델(임베딩+디코더스택+Norm) 테스트
   python layer_unit_test_MoshiModel.py
   ```
   - 변환된 레이어 및 모델들에 대해 **PyTorch vs. Flax** 수치 비교 메시지가 출력됩니다.

3. **에러가 없다면**  
   - “mean diff = 0.00xxx” 형태로 오차가 나오며, 문제없이 변환되었음을 의미합니다.

## 앞으로의 작업

1. **모델 전체 변환**  
   - 현재는 기본 레이어 및 디코더(텍스트 기반) 단위 **검증 완료**  
   - 추후 *modeling_moshi.py*의 상위 모델(예: `MoshiForConditionalGeneration`) 전체를 Flax로 이식  

---

## 1-1) 이미 구현 완료("어느정도" 테스트/검증됨)

- **FlexibleLinear**  
  - PyTorch: `MoshiFlexibleLinearPT`  
  - Flax: `MoshiFlexibleLinearFL`  

- **RMSNorm**  
  - PyTorch: `MoshiRMSNormPT`  
  - Flax: `MoshiRMSNormFL`  

- **GatingMLP**  
  - PyTorch: `MoshiGatingMLPPT`  
  - Flax: `MoshiGatingMLPFL`  

- **Attention (MoshiAttention) & RoPE**  
  - PyTorch: `MoshiAttentionPT`  
  - Flax: `MoshiAttentionFL`  
  - RotaryEmbedding 역시 PT↔Flax 매핑 확인됨  

- **DecoderLayer & MoshiModel**  
  - PyTorch: `MoshiDecoderLayerPT/MoshiModelPT`  
  - Flax: `MoshiDecoderLayerFL/MoshiModelFL`  
  - `layer_unit_test_MoshiDecoderLayer.py`, `layer_unit_test_MoshiModel.py`에서 PyTorch↔Flax 비교

---

## 1-2) 추가 구현(필수)

**(A) Depth Decoder**  
- **`MoshiDepthDecoder`**  
  - 오디오 전용 디코더 (Attention, GatingMLP 등 유사 구조)  
  - Flax 버전: `MoshiDepthDecoderFL`  
  - 추가 변환 및 PyTorch↔Flax 단위 테스트 필요

**(B) 최상위 클래스**  
- **`MoshiForConditionalGeneration`**  
  - 텍스트+오디오 결합, `depth_decoder`와 `decoder`를 결합  
  - Flax 버전(`MoshiForConditionalGenerationFL`)으로 전체 forward 로직, text/audio labels → loss 계산 등 구현  
  - PyTorch→Flax 가중치 로드 후 최종 diff 확인

---

## 1-3) 선택적(검토 필요)

### 1-3-1. Audio Encoder
- 예: `AutoModel`(PyTorch)
- Flax로 완전 이식하거나, PyTorch만 사용(“freeze”) 후 중간 산출물(코드 등)을 넘길지 여부 결정

### 1-3-2. Generate 함수
- PyTorch `generate(...)` → Flax에도 동일 “beam search” 등 구현할지
- 필요 없다면 skip하거나 minimal만 작성

### 1-3-3. PreTrainedModel 베이스
- PyTorch: `MoshiPreTrainedModel` 상속
- Flax: `FlaxPreTrainedModel` 유사 구조 구현할지 / 커스텀할지

---

**모듈 변환 이후 부가적 사항**

2. **Flax 학습 루프 구성**  
   - TrainState(Optax 등)으로 TPU에서 학습  
   - 멀티코어(pmap), jit 등 분산/컴파일 로직 적용  

3. **테스트 & 디버깅**     
   - float16 / bfloat16 등 TPU-friendly dtype 확인  

---

**문의나 개선사항**은 이슈로 등록해 주시면 감사하겠습니다. (Ch-talk-SKM)
