# Moshi Custom Layers: PyTorch → Flax/JAX for TPU

이 저장소는 **`modeling_moshi.py`**의 핵심 커스텀 레이어들을 **TPU 최적화**를 위해 **PyTorch에서 Flax/JAX로 변환**하고, 두 버전이 **동일하게 동작**하는지 확인하기 위한 단위 테스트 및 **기본 학습 스크립트**를 담고 있습니다.

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

- **`layer_unit_test_TPU_compare3.py`**  
  - FlexibleLinear, RMSNorm, GatingMLP 등 개별 레이어 단위 기본 테스트  
- **`layer_unit_test_TPU_att_rope_compare3.py`**  
  - MoshiAttention, RoPE(회전 위치 인코딩) 관련 PyTorch↔Flax 비교  
- **`layer_unit_test_MoshiDecoderLayer.py`**  
  - **`MoshiDecoderLayerPT`** vs **`MoshiDecoderLayerFL`** 구현 확인  
  - RMSNorm, GatingMLP, Self-Attention, RoPE 로직을 조합한 **디코더 레이어** 단위의 PyTorch↔Flax 매핑 테스트  
  - Flexible Linear 및 레이어 인덱스(layer_idx) 등을 통해 코드북 사용 시나리오 검증  
- **`layer_unit_test_MoshiModel.py`**  
  - **`MoshiModelPT`** vs **`MoshiModelFL`** (임베딩 + 다중 디코더 레이어 + 최종 Norm) 구조 확인  
  - Flexible Linear, Codebook 개수 설정 등 다양한 경우(토큰별 다른 codebook 인덱스 포함)에서 PyTorch↔Flax의 출력 정밀도 확인
- **`layer_unit_test_DepthDecoder.py`**  
  - **`MoshiDepthDecoderPT`** vs **`MoshiDepthDecoderFL`**를 테스트  
  - 오디오 전용 디코더 구조에 대한 PyTorch ↔ Flax 매핑 검증 및 layer_idx 동작 확인  

### Flax 기반 학습 스크립트

- **`train_model.py`**  
  - Flax/JAX로 구현된 간단한 **학습** 예시 코드  
  - **단일 디바이스 ** (TODO: 멀티 디바이스(pmap) 상황에서 GPU 1개 상황에서 에러..)
  - 배치 사이즈가 나누어떨어지지 않을 때 leftover를 **dummy 샘플**로 패딩해 shape 불일치를 방지  
  - `attention_mask`가 (B, S) 형태로 들어올 경우, (B, 1, 1, S)로 확장하여 (B, nHeads, S, S)와 브로드캐스팅이 가능하도록 처리  
  - `cross_entropy` 기반 간단 손실 함수와 AdamW 옵티마이저, Orbax를 이용한 체크포인트 로직 포함  
  - 향후 **gradient_accumulation**, **lr_scheduler**(warmup, decay 등), **bfloat16 최적화**, **beam search** 등의 확장 가능성을 남김  

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

   # (추가) DepthDecoder(오디오 전용) 테스트
   python layer_unit_test_DepthDecoder.py
   ```
   - 변환된 레이어 및 모델들에 대해 **PyTorch vs. Flax** 수치 비교 메시지가 출력됩니다.

3. **Flax 학습 스크립트(train_model.py) 실행**  
   - **단일 디바이스**(GPU) 환경에서 동작  
   - `batch_size`, `dataset_size`, `num_train_epochs`, `dtype(bf16/fp16)`, `swap_moshi_user` 등 하이퍼파라미터 조정 가능  
   - leftover 배치 자동 패딩, attention mask 확장 등의 로직 확인  
   ```bash
   python train_model.py --dataset_size 40 --batch_size 16 --max_length 256 \
       --num_train_epochs 5 --dtype bf16 --learning_rate 1e-4
   ```
   - Orbax 체크포인트를 통해 주기적으로 `TrainState` 저장 및 복원 가능  

4. **에러가 없다면**  
   - “mean diff = 0.00xxx” 형태로 테스트 스크립트에서 오차가 나오거나, train_model.py에서는 학습이 정상 수행됨  

## 앞으로의 작업

1. **모델 전체 변환**  
   - 현재는 기본 레이어, 텍스트 디코더(`MoshiDecoderLayer`) 및 오디오 디코더(`MoshiDepthDecoder`) 단위까지 **검증 완료**  
   - 추후 *modeling_moshi.py*의 상위 모델(예: `MoshiForConditionalGeneration`) 전체를 Flax로 이식  

2. **Flax 학습 루프 구성**  
   - 현재 `train_model.py`는 단순 로직 예시  
   - 추후 **gradient_accumulation_steps**, **lr_scheduler**(warmup, decay) 추가  
   - 멀티코어(pmap), jit 등 분산/컴파일 로직 확장 + 대규모 데이터셋에서 실제 성능 검증  

3. **테스트 & 디버깅**     
   - float16 / bfloat16 등 TPU-friendly dtype 확인  
   - 체크포인트 크기 문제 시 **orbax async checkpoint** 등 최적화 고려  

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

- **DepthDecoder**  
  - PyTorch: `MoshiDepthDecoderPT`  
  - Flax: `MoshiDepthDecoderFL`  
  - `layer_unit_test_DepthDecoder.py`에서 PyTorch↔Flax 비교

---

## 1-2) 추가 구현(필수)

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

**문의나 개선사항**은 이슈로 등록해 주시면 감사하겠습니다. (Ch-talk-SKM)
