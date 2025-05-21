

##  핵심 모듈별 “구현 상태” 체크리스트 (총 18 엔티티)

| #  | 모듈 (Upstream 이름)                  | Upstream 폴더               | **기능**                         | **TPU 포팅 현황**                   | 미구현·결함                        |
| -- | --------------------------------- | ------------------------- | ------------------------------ | ------------------------------- | ----------------------------- |
| 1  | **FlexibleLinear**                | `moshi/modeling_torch.py` | 채널별 코드북-선택 선형층                 | PT+Flax 구현, unit-test 통과        | `jax.precision` 옵션 없음         |
| 2  | **RMSNorm**                       | same                      | 스케일노름                          | PT+Flax 구현                      | dtype cast/eps config 불완전     |
| 3  | **GatedMLP**                      | same                      | GEGLU feed-forward             | 구현 완료                           | no‐gate fallback 없음           |
| 4  | **RotaryEmbedding**               | same                      | RoPE θ 스케일·동적 cache            | 구현 완료                           | Flax 쪽 `precision=HIGHEST` 누락 |
| 5  | **MultiHeadAttention**            | same                      | SDPA with key/value cache      | PT+Flax 버전 있으나 **Flash-Attn X** |                               |
| 6  | **DecoderLayer**                  | same                      | (Attn+MLP+Norm) block          | PT+Flax unit-test 통과            | —                             |
| 7  | **DepthDecoder**                  | same                      | codebook 간 종단간 의존 모델           | PT+Flax test OK                 | 실제 학습·loss curve 검증 미완        |
| 8  | **TemporalTransformer 7B**        | same                      | 1 k layer GPT-style backbone   | **미구현**                         | 7 B param init & ckpt load    |
| 9  | **InnerMonologueTextHead**        | same                      | 텍스트 토큰 보조 LM head              | 없음                              | Loss weight 스케줄 포함해야          |
| 10 | **MoshiForConditionalGeneration** | same                      | HF `generate()` wrapper        | 없음                              | Beam search, logits mask      |
| 11 | **MimiCodec (Rust)**              | `rust/`, `audio/`         | 24 kHz stream codec → tokens   | 전무                              | PyO3 wheel → XLA FFI          |
| 12 | **prepare\_dataset.py**           | `scripts/`                | wav→npz 배치 토크나이저               | 없음                              | Shard index, bucketing        |
| 13 | **train.py**                      | `scripts/`                | DeepSpeed, Flash-Attn8, ZeRO-3 | Flax 단일-core 예시만                | pjit mesh / grad accum        |
| 14 | **inference\_server.py**          | `client/`                 | WebSocket duplex, VAD          | 없음                              | Opus/PCM stream, beam sync    |
| 15 | **MLX variant**                   | `moshi_mlx/`              | iOS Metal backend              | 없음                              | 선택 사항                         |
| 16 | **CI workflow**                   | `.github/workflows`       | ruff, pytest, rust-fmt         | 없음                              | CPU jaxlib, torch compile     |
| 17 | **docs site**                     | docs GitHub Pages         | API + paper figures            | 없음                              | Sphinx-autodoc                |
| 18 | **Docker compose**                | root                      | GPU demo stack                 | 없음                              | TPU VM Dockerfile 작성          |

---

### 3  세부 갭 해설 (예시)

* **Codec ↔ Model 타이밍 맞춤**
  Mimi는 **80 ms 프레임(12.5 Hz)** 출력이므로, DepthDecoder 1 step == 1 frame임을 보장해야 함. TPU 포팅분엔 코덱이 없어 시간축 alignment 검증 자체가 불가능하다.
* **Flash-Attention in XLA**
  Upstream은 `FlashAttention2` CUDA를 `torchscale` fork로 호출; TPU에서는 `jax.lax.dot_general` rewire or `xla.call_tf_kernel("flash_attention", …)` 로 대체 구현 필요.
* **Distributed pjit Mesh**
  최적 샤딩: `('dp','fsdp')` → (`dp` = host-level data-parallel, `fsdp` = layer-wise param shards). `ShardingSpec` 미설정 시 XLA 메모리 터지므로 필수.


---


## 1. `layer_unit_test_TPU.py` 

| 범주             | 세부 내용                                                                                                                                                                                        |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **주요 정의**      | `MoshiFlexibleLinearPT/FL`, `MoshiRMSNormPT/FL`, `MoshiGatingMLPPT/FL` ― **PT ↔ Flax 1 : 1 대응**                                                                                              |
| **입력 → 출력 규칙** | `layer_idx` 인자의 3-way 모드<br>① `None` → 토큰축 S 가 곧 codebook 축 (*S == num\_layers*)<br>② `int` → 모든 토큰이 동일 코드북 사용<br>③ `1-D tensor/array` → token-wise codebook 선택                              |
| **검증 로직**      | (아래 두 함수가 파일 끝에 존재하지만 raw 본문이 1 줄로 반환돼 위치 인덱싱은 불가) <br>‣ `copy_pt_weights_to_flax()` : state-dict → FrozenDict 변환 <br>‣ `run_single_case()` : 난수 입력 생성 → 두 프레임워크 forward → MAE ≤ 1e-6 assert |
| **디자인 포인트**    | *FlexibleLinear* 에서 (B,S,D) ↔ (B,1,D) 자동 브로드캐스트 / `matmul_2d` einsum == `torch.nn.functional.linear` 대칭                                                                                      |
| **발견된 문제**     | ① 파일에 클래스 + 테스트 코드가 함께 들어 있어 **재사용 어려움**<br>② 함수·클래스 Docstring 은 있지만 **type hint 없음**<br>③ Flax 버전의 경우 XLA‐friendly einsum 이지만 `precision=lax.Precision.HIGHEST` 같은 옵션 미지정                   |

---

## 2. `layer_unit_test_TPU_compare3.py` 

| 범주          | 세부 내용                                                                                                                                                          |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **목적**      | **“원본 PyTorch”(`*_PT_origin`) vs 새 PyTorch(`*_PT`) vs Flax(`*_FL`)** ––– 세 변종을 한꺼번에 비교                                                                         |
| **핵심 유틸**   | `compare_two_model_outputs()` (PT ↔ PT) · `compare_pt_fl()` (PT ↔ FL) ― 둘 다 MAE 출력                                                                             |
| **중요 구현**   | *PT\_origin* 클래스들은 **HF `modeling_moshi.py` 원형** 일부를 그대로 옮긴 것. Flexible-Linear 가 `forward` 에서 **int · 1-D · None** 케이스를 모두 지원하되 *shape 불일치 시 에러* 를 명시적으로 raise |
| **특이점**     | 동일 파일 안에 **동형 클래스가 6 개**(PT\_origin/PT/FL × 3) → DRY 위배, 차후 패키지화 시 통폐합 필요                                                                                      |
| **로직 안전장치** | MAE 계산이 `np.abs(diff).mean()` 하나뿐 → Max diff 검증이나 dtype-cast 오류 탐지는 없음                                                                                         |

---

## 3. `layer_unit_test_TPU_att_rope.py` 

| 범주              | 세부 내용                                                                                                                                                           |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **추가 레이어**      | `MoshiRotaryEmbeddingPT`, `MoshiLinearPT`(표준 vs Flexible wrapper), `apply_rotary_pos_emb` 헬퍼                                                                    |
| **RoPE 초기화 전략** | 딕셔너리 `ROPE_INIT_FUNCTIONS`<br> • `default` : Θ = 10 000 <br> • `dynamic` : Θ = `config.rope_theta`, scaling ∝ `seq_len^-0.05`                                   |
| **주의 깊게 본 부분**  | `MoshiRotaryEmbeddingPT._dynamic_frequency_update()` : **position IDs가 기존 cache 길이 초과** 시 inv-freq 재계산 → **줄어든 길이**(fine-tune short seq) 에서는 *초기 inv-freq 로 롤백* |
| **결합 테스트**      | 아직 Attention 자체는 포함 안 됨 → RoPE 값만 검증(코사인·사인 테이블)                                                                                                                |

---

## 4. `layer_unit_test_TPU_att_rope_compare3.py` 

> “3-way compare” 버전의 **Attention + RoPE**. PT\_origin ↔ PT ↔ FL 모두 생성하고, `use_flexible_linear` 플래그에 따라 동일 weight copy 구간이 달라진다.
> *Flax 쪽 Attention* 은 `MoshiAttentionFL` 로 이미 포함돼 있으나, 파일 앞부분 주석에만 명시될 뿐 raw 본문은 길어서 여기 인용엔 노출 X.

**관찰**

* 세 버전 모두 **RoPE θ·head\_dim·codebook 수** 를 랜덤해시 세트 3-개씩 반복 테스트 → CI 자동화 시 매트릭스화 적합.
* PT → FL weight copy 함수는 `jax.tree_util.tree_map(lambda p: jnp.array(...))` 패턴.

---

## 5. `layer_unit_test_MoshiDecoderLayer.py` 

| 범주           | 세부 내용                                                                                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **구성요소**     | `MoshiDecoderLayerPT` ⇔ `MoshiDecoderLayerFL` (후자는 파일 하단에 위치)<br>내부 모듈 : Attention, Gating MLP, RMSNorm 모두 **전 파일에서 import**                                                          |
| **테스트 시나리오** | ‣ `use_flexible_linear` bool<br>‣ RoPE on/off<br>‣ codebook 인덱스 (None / int / 1-D) 세팅별 forward 검증                                                                                     |
| **버그 가능성**   | PT forward 서명은 `hidden_states, attention_mask=None, position_ids=None …`<br>FL 쪽 `__call__` 은 같은 순서지만 **키워드 인자 강제** → 테스트 함수에서 이름 순서 실수 시 Silent fail(**No error, wrong mapping**) 위험 |

---

## 6. `layer_unit_test_DepthDecoder.py` 

| 범주                        | 세부 내용                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Depth = N-layer 반복 구조** | `setup()` 루틴에서 `setattr(self, f"layer_{i}", MoshiDecoderLayerFL(...))` 동적 등록                                                           |
| **I/O 경로**                | (1) text 토큰 임베딩 → (B,1,H) <br>(2) audio 토큰 여러 개 → 0-패딩 + 선택 임베딩 (코드 참조) <br>(3) `input_linear` 로 `last_hidden_state` 투영 후 residual add |
| **loss**                  | cross-entropy w/ mask (`labels == -100`)                                                                                               |
| **Param Copy**            | `load_depth_decoder_pt_to_flax()` : PT module → Flax params, `jax.random.PRNGKey` 로 빈 파라미터 트리 생성 후 leaf 단위 replace                     |
| **리뷰 포인트**                | `embed_tokens` 의 첫 codebook 인덱스만 사용(0) — > 다중 코드북 학습 시 확장 필요                                                                           |


---



아래 비교는 **(1) MOSHI\_forTPU 레포의 3 개 스크립트**
`layer_unit_test_MoshiModel.py`, `train_model.py`, `train_exp.py`
와 **(2) Kyutai 공식 Moshi 레포**(`kyutai-labs/moshi`)의 동등한 부분(= 전체 모델·훈련 체인)을 **기능·API·데이터 흐름** 단위로 대조한 것입니다.

---

## 1  `layer_unit_test_MoshiModel.py` — Stub vs 완제품

| 항목         | MOSHI\_forTPU (파일 현황)                                                                                                      | Upstream Moshi (동등 기능)                                                                                                      | 차이 & 해야 할 일                                                                                                 |
| ---------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **존재 목적**  | “PyTorch ↔ Flax *전체 모델* 동치 검증”이라는 *스텁 파일*. 실제 코드는<br>`python<br># TODO Case A/B/C …<br>pass<br>` 정도만 있고 실행 시 아무 테스트도 돌지 않음 | Upstream에는 이 수준의 *test stub* 이 없다. 대신 **완전한 7 B 파라미터 모델**(PyTorch) 자체를 제공하며, 통합 테스트는 `pytest` 모듈로 모델 forward + loss 체크를 돌린다 | ① **Flax용 Moshi 7 B**(Temporal + Depth) 클래스를 먼저 완성<br>② PT → Flax 파라미터 복사 함수 작성<br>③ 이 파일을 `pytest` 스위트로 교체 |
| **Config** | `DummyConfig` 클래스만 스케치(4-5 속성)                                                                                             | `MoshiConfig` (`hidden_size`, `n_heads`, `codebooks`, …)                                                                    | Kyutai 스키마에 맞춘 **Flax Config dataclass** 신설 필요                                                              |
| **검증 범위**  | 없음 → 모든 assert 패스                                                                                                          | GPT-style backbone + DepthDecoder + Dual Audio/Text Heads 모두 커버                                                             | “Depth <-> Temporal 인터페이스” shape, RoPE θ, codebook-routing diff ≤ 1e-6 검증 추가                                |
| **결론**     | **컨셉 문서 수준**. 사실상 “아직 아무것도 구현 안 됨” —> 본격 테스트를 쓰려면 *실제 모델 정의 + 로더 + 파라미터 트리* 부터 작성해야 함                                      |                                                                                                                             |                                                                                                             |

---

## 2  학습 스크립트 비교 (Flax vs PyTorch)

### 2-A  `train_model.py` (Flax, TPU 용 단일 코어)

| 항목               | MOSHI\_forTPU 구현                                         | Upstream 대응 (`scripts/train.py`)            | Gap / 구현 필요                                      |
| ---------------- | -------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------ |
| **프레임워크 & 디바이스** | JAX/Flax, TPU *1 core only*.<br>`--pmap` 주석만 있고 실제 호출 없음 | PyTorch + DeepSpeed ZeRO-3, GPU 다중 노드       | TPU 멀티코어: `pjit` + `mesh` 샤딩, XLA bf16 추론        |
| **데이터 로더**       | `get_dummy_batch()` — 난수 토큰 생성                           | TFRecord / webdataset ⇢ **Mimi codec 토큰**   | Mimi codec wheel 빌드 + `tf.data` 로더 통합            |
| **손실 함수**        | 단일 `cross_entropy` (오디오 토큰)                              | **이중 손실**: ① 음성 토큰, ② 텍스트 토큰(내적 모놀로그), λ 조합 | Text prefix loss·λ 스케줄 추가                        |
| **옵티마이저**        | Optax AdamW (고정 LR)                                      | AdamW + `cosine_with_warmup` 스케줄            | `optax.warmup_cosine_decay_schedule`, grad accum |
| **체크포인트**        | Orbax (local)                                            | DeepSpeed sharded + WandB resume            | Orbax ↔ HF `save_pretrained` 브리지                 |
| **결론**           | “Flax 처음 손대본 PoC 루프”                                     | “실전 7 B 다중 GPU 학습 루프”                       | LR 스케줄·mixed bf16·분산 ShardSpec 필수                |

### 2-B  `train_exp.py` (PyTorch + 🤗 Accelerate)

| 항목                  | MOSHI\_forTPU                                             | Upstream                                         | Gap                             |
| ------------------- | --------------------------------------------------------- | ------------------------------------------------ | ------------------------------- |
| **런처**              | `accelerate launch`, DDP 1-8 GPU                          | DeepSpeed CLI(`--deepspeed_config`)              | ZeRO-stage, CPU offload 미지원     |
| **모델 로드**           | `MoshiDepthDecoderFL` 만                                   | `MoshiForConditionalGeneration` 7 B              | Temporal backbone·Dual head 전무  |
| **Mixed precision** | Amp bf16 off by default                                   | `torch.bfloat16` native, Flash-Attention kernels | Flash-Attention2, AMP 스케줄링      |
| **결론**              | “GPU 용 가벼운 예제” 수준                                         | 대규모 디바이스·ZeRO 최적화                                | Mimi 데이터 + 7 B 모델 맞춰 전면 리베이스 필요 |

---

| 파일                        | 핵심 내용                                        | 상태                                  |
| ------------------------- | -------------------------------------------- | ----------------------------------- |
| **`Readme.md`**           | 레포 목적·대상 레이어 5종·테스트 전략·TODO 요약               | 초안 수준 (세부 API doc 없음) ([GitHub][1]) |
| **`Note_att_rope.md`**    | RoPE “4 vs 2 broadcast” 오류 원인·수정법·수학적 정당성 정리 | 내용 충실, 코드 스니펫까지 포함 ([GitHub][2])    |
| **`train_model_todo.md`** | 학습 파이프라인 확장 체크리스트                            | 간단 체크리스트만 존재                        |

[1]: https://github.com/Ch-talk-SKM/MOSHI_forTPU "GitHub - Ch-talk-SKM/MOSHI_forTPU"
[2]: https://github.com/Ch-talk-SKM/MOSHI_forTPU/raw/main/Note_att_rope.md?plain=1 "github.com"


