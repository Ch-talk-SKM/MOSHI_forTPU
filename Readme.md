# Moshi Custom Layers: PyTorch → Flax/JAX for TPU

이 저장소는 **`modeling_moshi.py`**의 핵심 커스텀 레이어들을 **TPU 최적화**를 위해 **PyTorch에서 Flax/JAX로 변환**하고, 두 버전이 **동일하게 동작**하는지 확인하기 위한 단위 테스트를 담고 있습니다.

## 주요 배경

- TPU 환경에서 **JAX/Flax**가 PyTorch 대비 높은 효율을 내는 경우가 많아, **커스텀 연산 레이어**를 우선 변환 후 검증 중입니다.  
- 우선 변환 대상 커스텀 레이어:
  1. **MoshiFlexibleLinear** : https://github.com/huggingface/transformers/blob/main/src/transformers/models/moshi/modeling_moshi.py#L253  
  2. **MoshiRMSNorm** : https://github.com/huggingface/transformers/blob/main/src/transformers/models/moshi/modeling_moshi.py#L231
  3. **MoshiGatingMLP** : https://github.com/huggingface/transformers/blob/main/src/transformers/models/moshi/modeling_moshi.py#L408
  4. **MoshiRotaryEmbedding** : https://github.com/huggingface/transformers/blob/main/src/transformers/models/moshi/modeling_moshi.py#L311
  5. **MoshiAttention** : https://github.com/huggingface/transformers/blob/main/src/transformers/models/moshi/modeling_moshi.py#L446

## 현재 구성

1. **`class *PT`**: PyTorch 구현체
2. **`class *FL.py`**: Flax/JAX 구현체
3. **가중치 복사 & 결과 비교**  
   - PyTorch 레이어의 weight를 Flax로 로드 → **forward** 출력 비교(평균 차이 등)

## 사용 방법

1. Python 환경 준비  
   - PyTorch, JAX, Flax, NumPy 설치  
   - TPU 환경이라면 TPU용 jaxlib 준비

2. 테스트 스크립트 실행
   ```bash
   python layer_unit_test_TPU.py
   python layer_unit_test_TPU_att_rope.py
   ```
   - 변환된 레이어들에 대해 **PyTorch vs. Flax** 수치 비교 메시지가 출력됩니다.

3. 에러가 없다면  
   - “PyTorch vs. JAX average diff = 0.00xxx” 형태로 오차가 나오며, 문제없이 변환되었음을 의미

## 앞으로의 작업

1. **모델 전체 변환**  
   - 현재는 기본 레이어 단위 **검증 완료**
   - 추후 *modeling_moshi.py*의 상위 모델(예: MoshiForConditionalGeneration) 전체를 Flax로 이식

---
## 1-1) 이미 구현 완료("어느정도" 테스트/검증됨)

- **FlexibleLinear**  
  - PyTorch: `MoshiFlexibleLinearPT` / `_PT_origin`  
  - Flax: `MoshiFlexibleLinearFL`  
- **RMSNorm**  
  - PyTorch: `MoshiRMSNormPT` / `_PT_origin`  
  - Flax: `MoshiRMSNormFL`  
- **GatingMLP**  
  - PyTorch: `MoshiGatingMLPPT` / `_PT_origin`  
  - Flax: `MoshiGatingMLPFL`  
- **Attention (MoshiAttention)**  
  - PyTorch: `MoshiAttentionPT` / `_PT_origin`  
  - Flax: `MoshiAttentionFL`  
- **RotaryEmbedding (RoPE)**  
  - PyTorch: `MoshiRotaryEmbeddingPT` / `_PT_origin`  
  - Flax: `MoshiRotaryEmbeddingFL`

이들은 이미 `layer_unit_test_TPU*.py` 계열 테스트 스크립트에서 **PyTorch ↔ Flax** 매핑이 검증됨.

---

## 1-2) 추가 구현(필수)

**(A) Decoder 레이어 & Model**  
- **`MoshiDecoderLayer`**  
  - 내부에서 Attention, RMSNorm, GatingMLP 등 조합 → Flax 버전(`MoshiDecoderLayerFL`) 작성  
- **`MoshiModel`** (텍스트 디코더)  
  - 위 레이어 n개 스택, 임베딩, final norm 등 → `MoshiModelFL`

**(B) Depth Decoder**  
- **`MoshiDepthDecoder`**  
  - 오디오 전용 디코더 (Attention, GatingMLP, etc. 유사구조) → `MoshiDepthDecoderFL`  

**(C) 최상위 클래스**  
- **`MoshiForConditionalGeneration`**  
  - 텍스트+오디오 결합, `depth_decoder`와 `decoder`를 결합  
  - Flax 버전: `MoshiForConditionalGenerationFL`  
  - 전체 `forward(...)` 로직, text/audio labels → loss 계산 등

(필수 구현이 끝나면, PyTorch→Flax 가중치 로드를 마치고 최종 diff 확인)

---

## 1-3) 선택적(검토 필요)

1-3-1. **Audio Encoder**  
   - 예: `AutoModel`(PyTorch)  
   - Flax로 완전 이식 or PyTorch만 사용(“freeze”) 후 “codes”를 넘길지 결정  
   - 프로젝트 상황에 따라 선택

1-3-2. **Generate 함수**  
   - PyTorch `generate(...)` → Flax에도 동일 “beam search” 등 구현할지,  
   - 필요 없다면 skip하거나 minimal만 작성

1-3-3. **PreTrainedModel 베이스**  
   - PyTorch: `MoshiPreTrainedModel` 상속  
   - Flax: `FlaxPreTrainedModel` 비슷한 구조 구현할지 / 커스텀할지

---

**모듈 변환 이후 부가적 사항**

2. **Flax 학습 루프 구성**  
   - TrainState(Optax 등)으로 TPU에서 학습  
   - 멀티코어(pmap), jit 등 분산/컴파일 로직 적용

3. **테스트 & 디버깅**     
   - float16 / bfloat16 등 TPU-friendly dtype 확인

---

**문의나 개선사항**은 이슈로 등록해 주시면 감사하겠습니다. (Ch-talk-SKM)
