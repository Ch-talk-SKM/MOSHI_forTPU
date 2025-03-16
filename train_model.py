#!/usr/bin/env python3
# coding=utf-8

"""
Flax training code with leftover padding (no partial batches), single-vs-multi device logic,
and automatic attention_mask dimension expansion for broadcast with (B,nHeads,S,S).

- Single device => uses single-device training (no pmap).
- Multiple devices => uses pmap for data parallel.
- Leftover is padded with dummy samples to avoid shape=().
- We expand (B,S) attention_mask -> (B,1,1,S) to avoid broadcast errors.

향후 해야할것들 (예시):
1) 실제 모델 구조에 맞춰 rope scaling/position_ids/logits 등 더욱 세부 조정 필요.
2) gradient_accumulation_steps 로직 추가 (현재 단순 예시).
3) lr_scheduler (warmup, decay 등) 도입.
4) checkpoint가 커질 경우 orbax async checkpoint 등 최적화 고려.
5) GPU/TPU 환경(멀티 디바이스)에서 실제 성능·안정성 테스트.
"""

import argparse
import os
from functools import partial
from typing import Any, Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

# 가정: 아래 모듈들은 이미 구현되어 있음
from train_exp import (
    chunk_example,
    swap_moshi_user,
    MoshiDataCollator,
)
from layer_unit_test_MoshiModel import (
    MoshiModelFL,
    DummyMoshiConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Flax training with leftover padding & single/multi device logic.")
    parser.add_argument("--dataset_size", type=int, default=40)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--swap_moshi_user", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--audio_vocab_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--output_dir", type=str, default="flax_ckpts")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    return parser.parse_args()


class TrainState(train_state.TrainState):
    """기본 Flax TrainState (optimizer & params)."""
    pass


def create_dummy_dataset(dataset_size: int, vocab_size: int, audio_vocab_size: int):
    """
    (1) 충분히 큰 seq_len을 가진 더미 샘플 생성
    (2) chunk_example에서 빈 chunk가 발생하지 않도록 >=300 길이로 맞춤
    """
    rng = np.random.default_rng(0)
    data = {
        "moshi_text_tokens": [],
        "moshi_audio_tokens": [],
        "user_text_tokens": [],
        "user_audio_tokens": [],
    }
    for _ in range(dataset_size):
        seq_len = rng.integers(300, 600)
        moshi_text = rng.integers(0, vocab_size, size=(seq_len,))
        user_text  = rng.integers(0, vocab_size, size=(seq_len,))
        moshi_audio= rng.integers(0, audio_vocab_size, size=(8, seq_len))
        user_audio = rng.integers(0, audio_vocab_size, size=(8, seq_len))

        data["moshi_text_tokens"].append(moshi_text)
        data["moshi_audio_tokens"].append(moshi_audio)
        data["user_text_tokens"].append(user_text)
        data["user_audio_tokens"].append(user_audio)

    print(f"Dataset generated: {dataset_size} samples, first sample text len={len(data['moshi_text_tokens'][0])}")
    return data


def create_model_and_state(args):
    """
    MoshiModelFL 인스턴스 + 초기 파라미터(AdamW) 생성.
    """
    config = DummyMoshiConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        head_dim=args.head_dim,
        vocab_size=args.vocab_size,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        rope_scaling=None,
        use_flexible_linear=False,
        num_codebooks=1,
        ffn_dim=args.hidden_size*4,
        hidden_act="gelu",
        rms_norm_eps=1e-6,
    )
    model = MoshiModelFL(config=config)

    # dummy input for init
    rng = jax.random.PRNGKey(args.seed)
    sample_ids  = jnp.zeros((1,10), dtype=jnp.int32)
    sample_mask = jnp.ones((1,1,10,10), dtype=jnp.float32)
    init_vars   = model.init(rng, sample_ids, attention_mask=sample_mask)

    tx = optax.adamw(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    state = TrainState.create(
        apply_fn=model.apply,
        params=init_vars["params"],
        tx=tx
    )
    return state, model


def compute_loss(params, batch, model):
    """
    - attention_mask(ndim=2 => (B,S)) 이라면, (B,1,1,S)로 확장해 (B,nHeads,S,S)와 브로드캐스팅 호환
    - (B,1,1,S) or (B,nHeads,S,S) 등 4D 형태라면 그대로 사용
    - labels => cross-entropy
    """
    # 차원 자동 확장 로직
    attn_mask = batch["attention_mask"]
    if attn_mask.ndim == 2:
        attn_mask = attn_mask[:, None, None, :]  # => (B,1,1,S)

    outputs = model.apply({"params": params}, batch["input_ids"], attention_mask=attn_mask)
    B,S,D = outputs.shape

    # cross-entropy
    labels = batch["text_labels"]
    oh     = jax.nn.one_hot(labels, D)
    logp   = jax.nn.log_softmax(outputs, axis=-1)
    loss   = -jnp.sum(oh * logp, axis=-1).mean()
    return loss


@partial(jax.value_and_grad, has_aux=False)
def loss_fn(params, batch, model):
    return compute_loss(params, batch, model)


def train_step_single_device(state: TrainState, batch: Dict[str,jnp.ndarray], model: MoshiModelFL):
    """
    단일 디바이스 전용 => pmap 없이 jax.value_and_grad
    """
    def _loss_fn(p):
        return compute_loss(p, batch, model)
    loss, grads = jax.value_and_grad(_loss_fn)(state.params)

    g_norm = optax.global_norm(grads)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_state = state.replace(step=state.step+1, params=new_params, opt_state=new_opt_state)
    return new_state, loss, g_norm


@partial(jax.pmap, axis_name="batch")
def train_step_pmap(state: TrainState, batch: Dict[str,jnp.ndarray], model: MoshiModelFL):
    """
    멀티 디바이스 => pmap (data parallel)
    """
    def _loss_fn(p):
        return compute_loss(p, batch, model)

    loss, grads = jax.value_and_grad(_loss_fn)(state.params)
    # 장치 간 all-reduce
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss  = jax.lax.pmean(loss,  axis_name="batch")

    g_norm = optax.global_norm(grads)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_state = state.replace(step=state.step+1, params=new_params, opt_state=new_opt_state)
    return new_state, loss, g_norm


def shard_batch(batch: Dict[str, np.ndarray], num_devices:int)->Dict[str, np.ndarray]:
    """
    leftover 에러 방지를 위해 leftover=0이 되는 구조이지만, 여기서도 shape=() 체크
    """
    out = {}
    for k,v in batch.items():
        def _shard(x):
            return x.reshape((num_devices, -1) + x.shape[1:])
        out[k] = _shard(v)
    return out


def pad_leftover_batch(samples: list, needed: int, max_len=256, vocab_size=128, audio_vocab_size=16):
    """
    leftover 부분을 dummy 샘플(0으로 채움)로 채워 batch_size 일치
    """
    dummy_samples = []
    for _ in range(needed):
        dummy_moshi_text  = np.zeros((max_len,), dtype=np.int64)
        dummy_user_text   = np.zeros((max_len,),  dtype=np.int64)
        dummy_moshi_audio = np.zeros((8, max_len), dtype=np.int64)
        dummy_user_audio  = np.zeros((8, max_len), dtype=np.int64)

        dummy_samples.append({
            "moshi_text_tokens":  dummy_moshi_text,
            "user_text_tokens":   dummy_user_text,
            "moshi_audio_tokens": dummy_moshi_audio,
            "user_audio_tokens":  dummy_user_audio,
        })
    return samples + dummy_samples


def main():
    args = parse_args()

    # orbax 절대경로
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    num_devices = jax.device_count()
    print(f"=== JAX/Flax Training: num_devices={num_devices}, dtype={args.dtype} ===")

    # 멀티 디바이스 => pmap, 단일 디바이스 => single_device
    if num_devices > 1:
        print("[INFO] multi-device => pmap approach")
        train_step_func = train_step_pmap
        use_pmap = True
    else:
        print("[WARNING] single device => skip pmap => single-device training")
        train_step_func = train_step_single_device
        use_pmap = False

    # dtype 설정
    if args.dtype == "fp16":
        jax.experimental.enable_x64()
    # bf16이면 jax에서 bfloat16 자동
    # 나머지는 fp32

    # 1) 데이터 생성
    data_raw = create_dummy_dataset(args.dataset_size, args.vocab_size, args.audio_vocab_size)

    # optional swap
    if args.swap_moshi_user:
        data_raw = swap_moshi_user(data_raw)

    # 2) chunk
    chunked = chunk_example(data_raw, max_length=args.max_length)
    zipped  = []
    for (mt, ut, ma, ua) in zip(
        chunked["moshi_text_tokens"],
        chunked["user_text_tokens"],
        chunked["moshi_audio_tokens"],
        chunked["user_audio_tokens"],
    ):
        zipped.append({
            "moshi_text_tokens": mt,
            "user_text_tokens":  ut,
            "moshi_audio_tokens":ma,
            "user_audio_tokens": ua,
        })

    # 3) 모델 준비
    state, model = create_model_and_state(args)
    if use_pmap:
        state = jax.device_put_replicated(state, jax.devices())

    # 4) checkpoint
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt_opts    = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    ckpt_manager = ocp.CheckpointManager(
        directory=args.output_dir,
        checkpointers={"state": checkpointer},
        options=ckpt_opts
    )
    if args.resume_checkpoint:
        restored = ckpt_manager.restore(step=args.resume_checkpoint, items={"state": state})
        state = restored["state"]
        print(f"[INFO] resumed from step={args.resume_checkpoint}")

    # 5) collator (keys => input_ids, attention_mask, text_labels, etc.)
    class MyDataCollator(MoshiDataCollator):
        def __call__(self, features):
            base = super().__call__(features)
            # 여기서 base에는
            # 'input_ids', 'attention_mask', 'text_labels', 'audio_labels' 등이 포함됨
            return base

    collator = MyDataCollator(
        text_pad_token_id=args.vocab_size,
        audio_pad_token_id=args.audio_vocab_size,
        text_vocab_size=args.vocab_size,
        audio_vocab_size=args.audio_vocab_size,
        text_align_pad_token_id=0
    )

    # leftover => pad
    total_samples = len(zipped)
    full_steps    = total_samples // args.batch_size
    leftover      = total_samples %  args.batch_size
    step_count    = full_steps + (1 if leftover>0 else 0)

    # epoch loop
    global_step = 0
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.num_train_epochs):
        rng.shuffle(zipped)  # epoch별 shuffle

        for step_i in range(step_count):
            if step_i < full_steps:
                batch_samples = zipped[step_i*args.batch_size : (step_i+1)*args.batch_size]
            else:
                # leftover => pad
                leftover_samples = zipped[full_steps*args.batch_size : ]
                needed = args.batch_size - len(leftover_samples)
                if needed>0:
                    leftover_samples = pad_leftover_batch(
                        leftover_samples, needed,
                        max_len=args.max_length,
                        vocab_size=args.vocab_size,
                        audio_vocab_size=args.audio_vocab_size
                    )
                batch_samples = leftover_samples

            collated = collator(batch_samples)
            # torch=>np
            batch_np={}
            for k,v in collated.items():
                if hasattr(v,"numpy"):
                    v=v.numpy()
                batch_np[k]=v

            # pmap => shard
            if use_pmap:
                batch_shard = shard_batch(batch_np, jax.device_count())
                new_state, loss_val, grad_norm = train_step_func(state, batch_shard, model)
                # loss_val, grad_norm shape=(num_devices,)
                loss_scalar = float(loss_val[0])
                grad_scalar= float(grad_norm[0])
            else:
                new_state, loss_val, grad_norm = train_step_func(state, batch_np, model)
                loss_scalar = float(loss_val)
                grad_scalar= float(grad_norm)

            state = new_state
            global_step += 1

            # 10 step마다 loss 표시
            if global_step % 10 == 0:
                print(f"[Epoch {epoch} Step {global_step}] loss={loss_scalar:.4f}, grad_norm={grad_scalar:.4f}")

            if global_step % args.save_steps == 0:
                ckpt_manager.save(step=global_step, items={"state": state}, force=True)
                print(f"[INFO] checkpoint saved at step={global_step}")

    # 마지막 체크포인트
    ckpt_manager.save(step=global_step, items={"state": state}, force=True)
    print("Done training. Final checkpoint saved.")


if __name__=="__main__":
    main()
