# Moshi Custom Layers: PyTorch → Flax/JAX for TPU

이 저장소는 **`modeling_moshi.py`**의 핵심 커스텀 레이어들을 **TPU 최적화**를 위해 **PyTorch에서 Flax/JAX로 변환**하고, 두 버전이 **동일하게 동작**하는지 확인하기 위한 단위 테스트를 담고 있습니다.

## 주요 배경

- TPU 환경에서 **JAX/Flax**가 PyTorch 대비 높은 효율을 내는 경우가 많아, **커스텀 연산 레이어**를 우선 변환 후 검증 중입니다.  
- 우선 변환 대상 커스텀 레이어:
  1. **MoshiFlexibleLinear** : https://github.com/huggingface/transformers/blob/main/src/transformers/models/moshi/modeling_moshi.py#L253  
  2. **MoshiRMSNorm** : https://github.com/huggingface/transformers/blob/main/src/transformers/models/moshi/modeling_moshi.py#L231
  3. **MoshiGatingMLP** : https://github.com/huggingface/transformers/blob/main/src/transformers/models/moshi/modeling_moshi.py#L408

## 현재 구성

1. **`*_PT.py`**: PyTorch 구현체
2. **`*_FL.py`**: Flax/JAX 구현체
3. **가중치 복사 & 결과 비교**  
   - PyTorch 레이어의 weight를 Flax로 로드 → **forward** 출력 비교(평균 차이 등)

## 사용 방법

1. Python 환경 준비  
   - PyTorch, JAX, Flax, NumPy 설치  
   - TPU 환경이라면 TPU용 jaxlib 준비

2. 테스트 스크립트 실행
   ```bash
   python layer_unit_test_TPU.py
   ```
   - 변환된 레이어들에 대해 **PyTorch vs. Flax** 수치 비교 메시지가 출력됩니다.

3. 에러가 없다면  
   - “PyTorch vs. JAX average diff = 0.00xxx” 형태로 오차가 나오며, 문제없이 변환되었음을 의미

## 앞으로의 작업

1. **모델 전체 변환**  
   - 현재는 기본 레이어 단위 **검증 완료**
   - 추후 *modeling_moshi.py*의 상위 모델(예: MoshiForConditionalGeneration) 전체를 Flax로 이식

2. **Flax 학습 루프 구성**  
   - TrainState(Optax 등)으로 TPU에서 학습  
   - 멀티코어(pmap), jit 등 분산/컴파일 로직 적용

3. **테스트 & 디버깅**     
   - float16 / bfloat16 등 TPU-friendly dtype 확인

---

**문의나 개선사항**은 이슈로 등록해 주시면 감사하겠습니다. (Ch-talk-SKM)
