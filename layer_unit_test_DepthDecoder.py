#!/usr/bin/env python3
# coding=utf-8

import math
import copy
from dataclasses import dataclass
from typing import Optional, Union, Any

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

import torch
import torch.nn as pt_nn

# -----------------------------------------------------------
# (외부) 모듈 임포트 -- 실제 프로젝트에 맞춰 수정하거나 경로 조정 필요
# -----------------------------------------------------------
from layer_unit_test_TPU_compare3 import (
    MoshiFlexibleLinearPT,
    MoshiFlexibleLinearFL,
    load_pt_weights_flexlinear,
)
from layer_unit_test_MoshiDecoderLayer import (
    MoshiDecoderLayerPT,
    MoshiDecoderLayerFL,
    load_decoder_layer_pt_to_flax,
)
from layer_unit_test_TPU_att_rope_compare3 import (
    load_pytorch_weights_into_flax_attention,
)


@dataclass
class DummyDepthConfig:
    vocab_size: int=10
    audio_vocab_size: int=12
    hidden_size: int=8
    num_codebooks: int=3
    input_size: int=8
    num_hidden_layers: int=2

    num_attention_heads: int=2
    head_dim: int=4
    rope_theta: float=10000.0
    rope_scaling: dict=None
    ffn_dim: int=32
    hidden_act: str="gelu"
    rms_norm_eps: float=1e-6


# -----------------------------------------------------------
# 1) PyTorch DepthDecoder with flexible layer_idx
# -----------------------------------------------------------
class MoshiDepthDecoderPT(pt_nn.Module):
    """
    PyTorch 버전 MoshiDepthDecoder:
      - text_embed_tokens: (vocab_size+1, hidden_size)
      - embed_tokens: (num_codebooks-1)개의 오디오 임베딩
      - input_projections: MoshiFlexibleLinearPT
      - layers: MoshiDecoderLayerPT (num_hidden_layers개)
      - lm_heads: MoshiFlexibleLinearPT
    """
    def __init__(self, config: DummyDepthConfig):
        super().__init__()
        self.config = config
        self.text_embed_tokens = pt_nn.Embedding(config.vocab_size+1, config.hidden_size)
        self.embed_tokens = pt_nn.ModuleList([
            pt_nn.Embedding(config.audio_vocab_size+1, config.hidden_size)
            for _ in range(config.num_codebooks-1)
        ])
        self.input_projections = MoshiFlexibleLinearPT(
            config.input_size, config.hidden_size, config.num_codebooks
        )

        self.layers = pt_nn.ModuleList([
            MoshiDecoderLayerPT(config, layer_idx=i, use_flexible_linear=True, use_rope=False)
            for i in range(config.num_hidden_layers)
        ])

        self.lm_heads = MoshiFlexibleLinearPT(
            config.hidden_size, config.audio_vocab_size, config.num_codebooks
        )

    def forward(
        self,
        input_ids: torch.Tensor,          # (B,S)
        last_hidden_state: torch.Tensor,  # (B,S,input_size)
        layer_idx_input: Optional[Union[int, torch.Tensor]]=0,
        layer_idx_attn: Optional[Union[int, torch.Tensor]]=0,
        layer_idx_mlp:  Optional[Union[int, torch.Tensor]]=0,
        layer_idx_out:  Optional[Union[int, torch.Tensor]]=0,
        attention_mask: Optional[torch.Tensor]=None,
        labels: Optional[torch.Tensor]=None
    ):
        B,S = input_ids.shape

        # text embedding => first token
        text_ids = input_ids[:, 0]
        text_emb = self.text_embed_tokens(text_ids)  # (B, hidden_size)

        # audio embedding => 2nd token only, 나머지 토큰은 zero
        audio_emb_list = []
        for i in range(1, S):
            if i == 1:
                audio_emb_list.append(self.embed_tokens[0](input_ids[:, i]))
            else:
                audio_emb_list.append(torch.zeros_like(text_emb))
        if len(audio_emb_list) > 0:
            audio_emb = torch.stack(audio_emb_list, dim=1)  # (B, S-1, hidden_size)
        else:
            audio_emb = torch.zeros(B, 0, self.config.hidden_size, dtype=text_emb.dtype, device=text_emb.device)

        text_emb = text_emb.unsqueeze(1)               # (B,1,hidden_size)
        inputs_embeds = torch.cat([text_emb, audio_emb], dim=1)  # (B,S,hidden_size)

        # input projections
        proj_out = self.input_projections(last_hidden_state, layer_idx=layer_idx_input)
        hidden_states = inputs_embeds + proj_out  # (B,S,hidden_size)

        # pass through layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_idx_attn=layer_idx_attn,
                layer_idx_mlp=layer_idx_mlp
            )

        # final => lm_heads
        logits = self.lm_heads(hidden_states, layer_idx=layer_idx_out)

        loss = None
        if labels is not None:
            log_2d = logits.view(-1, self.config.audio_vocab_size)  # (B*S, audio_vocab_size)
            lab_1d = labels.view(-1)
            loss_fct = pt_nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(log_2d, lab_1d)

        return {"loss": loss, "logits": logits}


# -----------------------------------------------------------
# 2) Flax DepthDecoder with flexible layer_idx
# -----------------------------------------------------------
class MoshiDepthDecoderOutputFL:
    """단순한 Flax 출력 자료구조."""
    def __init__(self, loss=None, logits=None, hidden_states=None):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states

class MoshiDepthDecoderFL(nn.Module):
    """
    Flax 버전 MoshiDepthDecoder:
      - text_embed_tokens: param(shape=(vocab_size+1, hidden_size))
      - embed_tokens: param(shape=(num_codebooks-1, audio_vocab_size+1, hidden_size))
      - input_linear: MoshiFlexibleLinearFL
      - layer_0, layer_1, ... (num_hidden_layers개) => 각각 MoshiDecoderLayerFL
      - output_linear: MoshiFlexibleLinearFL
    """
    config: Any

    def setup(self):
        # 임베딩 파라미터
        self.text_embed_tokens = self.param(
            "text_embed_tokens",
            nn.initializers.normal(stddev=1.0),
            (self.config.vocab_size+1, self.config.hidden_size),
        )
        self.embed_tokens = self.param(
            "embed_tokens",
            nn.initializers.normal(stddev=1.0),
            (self.config.num_codebooks-1, self.config.audio_vocab_size+1, self.config.hidden_size),
        )

        # input projections
        self.input_linear = MoshiFlexibleLinearFL(
            self.config.input_size,
            self.config.hidden_size,
            self.config.num_codebooks,
            name="input_projections"
        )

        # num_hidden_layers 만큼 layer_i를 속성으로 등록
        for i in range(self.config.num_hidden_layers):
            setattr(
                self,
                f"layer_{i}",
                MoshiDecoderLayerFL(
                    config=self.config,
                    use_flexible_linear=True,
                    use_rope=False,
                    name=f"layer_{i}",
                )
            )

        # 최종 출력용
        self.output_linear = MoshiFlexibleLinearFL(
            self.config.hidden_size,
            self.config.audio_vocab_size,
            self.config.num_codebooks,
            name="lm_heads"
        )

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,          # (B,S)
        last_hidden_state: jnp.ndarray,  # (B,S,input_size)
        layer_idx_input: Union[int,jnp.ndarray,None]=0,
        layer_idx_attn: Union[int,jnp.ndarray,None]=0,
        layer_idx_mlp:  Union[int,jnp.ndarray,None]=0,
        layer_idx_out:  Union[int,jnp.ndarray,None]=0,
        attention_mask: Optional[jnp.ndarray]=None,
        labels: Optional[jnp.ndarray]=None
    ):
        B, S = input_ids.shape

        # 첫 토큰 => text 임베딩
        text_token = input_ids[:, 0]
        text_emb_vec = jnp.take(self.text_embed_tokens, text_token, axis=0)  # (B, hidden_size)

        # 두 번째 토큰 => audio 임베딩, 나머지는 0
        audio_emb_list = []
        for i in range(1, S):
            if i == 1:
                audio_emb_list.append(
                    jnp.take(self.embed_tokens[0], input_ids[:, i], axis=0)
                )  # (B, hidden_size)
            else:
                audio_emb_list.append(jnp.zeros_like(text_emb_vec))  # 0-fill
        if audio_emb_list:
            audio_emb = jnp.stack(audio_emb_list, axis=1)  # (B, S-1, hidden_size)
        else:
            audio_emb = jnp.zeros((B, 0, self.config.hidden_size), dtype=text_emb_vec.dtype)

        text_emb_vec = text_emb_vec[:, None, :]               # (B,1,hidden_size)
        inputs_embeds = jnp.concatenate([text_emb_vec, audio_emb], axis=1)  # (B,S,hidden_size)

        # input proj
        proj_out = self.input_linear(last_hidden_state, layer_idx=layer_idx_input)
        hidden_states = inputs_embeds + proj_out  # (B,S,hidden_size)

        # 레이어 순회
        for i in range(self.config.num_hidden_layers):
            lyr = getattr(self, f"layer_{i}")  # MoshiDecoderLayerFL
            hidden_states = lyr(
                hidden_states,
                attention_mask=attention_mask,
                layer_idx_attn=layer_idx_attn,
                layer_idx_mlp=layer_idx_mlp
            )

        # 최종 => output_linear
        logits = self.output_linear(hidden_states, layer_idx=layer_idx_out)

        # (옵션) loss
        loss = None
        if labels is not None:
            vocab_size = self.config.audio_vocab_size
            onehot = jax.nn.one_hot(labels, vocab_size)
            log_probs = nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            loss_per_token = -jnp.sum(onehot * log_probs, axis=-1)
            mask = (labels != -100).astype(logits.dtype)
            loss = jnp.sum(loss_per_token * mask) / jnp.sum(mask)

        return MoshiDepthDecoderOutputFL(loss, logits, hidden_states)


# -----------------------------------------------------------
# 3) PT->FL param load
# -----------------------------------------------------------
def load_depth_decoder_pt_to_flax(
    pt_depth: MoshiDepthDecoderPT,
    fl_depth: MoshiDepthDecoderFL,
    rng,
    sample_input_ids: torch.Tensor,
    sample_last_hidden: torch.Tensor,
):
    """
    PyTorch MoshiDepthDecoder -> Flax MoshiDepthDecoder 파라미터 복사.
    """
    # 1) 우선 Flax 모델 init으로 전체 파라미터 구조 생성
    init_vars = fl_depth.init(
        rng,
        jnp.array(sample_input_ids.cpu().numpy()),
        jnp.array(sample_last_hidden.cpu().numpy()),
        layer_idx_input=0,
        layer_idx_attn=0,
        layer_idx_mlp=0,
        layer_idx_out=0
    )
    params = copy.deepcopy(init_vars["params"])

    # 2) text_embed_tokens 복사
    pt_text_w = pt_depth.text_embed_tokens.weight.detach().cpu().numpy()  # (vocab_size+1, hidden_size)
    params["text_embed_tokens"] = jnp.array(pt_text_w)

    # 3) embed_tokens 복사 (PT에서는 ModuleList로 여러 임베딩, FL에서는 한 덩어리)
    embed_list = []
    for emb in pt_depth.embed_tokens:
        w_np = emb.weight.detach().cpu().numpy()  # (audio_vocab_size+1, hidden_size)
        embed_list.append(w_np)
    embed_np = np.stack(embed_list, axis=0)  # => shape (num_codebooks-1, audio_vocab_size+1, hidden_size)
    params["embed_tokens"] = jnp.array(embed_np)

    # 4) input_projections (MoshiFlexibleLinear) 복사
    dummy_inproj = MoshiFlexibleLinearFL(
        pt_depth.config.input_size,
        pt_depth.config.hidden_size,
        pt_depth.config.num_codebooks,
        name="dummy_inproj"
    )
    dummy_init_ip = dummy_inproj.init(
        rng,
        jnp.array(sample_last_hidden.cpu().numpy()),
        layer_idx=0
    )
    ip_vars = load_pt_weights_flexlinear(
        pt_depth.input_projections,
        dummy_inproj,
        rng,
        sample_last_hidden,
        layer_idx=0
    )
    params["input_projections"] = copy.deepcopy(ip_vars["params"])

    # 5) layers (각각 MoshiDecoderLayerPT -> MoshiDecoderLayerFL)
    for i, pt_layer in enumerate(pt_depth.layers):
        dummy_layer_fl = MoshiDecoderLayerFL(
            config=pt_layer.config, use_flexible_linear=True, use_rope=False, name=f"dummy_layer_{i}"
        )
        # 더미 init
        dummy_init_lyr = dummy_layer_fl.init(
            rng,
            jnp.array(sample_last_hidden.cpu().numpy()),
            layer_idx_attn=0,
            layer_idx_mlp=0
        )
        sub_init = load_decoder_layer_pt_to_flax(
            pt_layer,
            dummy_layer_fl,
            rng,
            sample_last_hidden
        )
        # 실제 Flax 모델에서는 "layer_{i}"라는 속성으로 등록 -> 파라미터 키도 "layer_{i}"
        layer_key = f"layer_{i}"  
        params[layer_key] = copy.deepcopy(sub_init["params"])

    # 6) lm_heads 복사
    dummy_lm_fl = MoshiFlexibleLinearFL(
        pt_depth.config.hidden_size,
        pt_depth.config.audio_vocab_size,
        pt_depth.config.num_codebooks,
        name="dummy_lmheads"
    )
    dummy_init_lm = dummy_lm_fl.init(
        rng,
        jnp.array(sample_last_hidden.cpu().numpy()),
        layer_idx=0
    )
    out_vars = load_pt_weights_flexlinear(
        pt_depth.lm_heads, dummy_lm_fl,
        rng, sample_last_hidden, layer_idx=0
    )
    params["lm_heads"] = copy.deepcopy(out_vars["params"])

    final_vars = copy.deepcopy(init_vars)
    final_vars["params"] = params
    return final_vars


# -----------------------------------------------------------
# 4) compare fns
# -----------------------------------------------------------
def compare_depth_decoder_pt_fl(
    pt_model: MoshiDepthDecoderPT,
    fl_model: MoshiDepthDecoderFL,
    fl_params,
    input_ids_pt: torch.Tensor,
    last_hidden_pt: torch.Tensor,
    layer_idx_in:  Union[int, torch.Tensor, None]=None,
    layer_idx_attn:Union[int, torch.Tensor, None]=None,
    layer_idx_mlp: Union[int, torch.Tensor, None]=None,
    layer_idx_out: Union[int, torch.Tensor, None]=None,
    labels_pt: Optional[torch.Tensor]=None,
    label=""
):
    """Compare PT vs FL forward with flexible layer_idx usage."""

    # 1) PyTorch forward
    with torch.no_grad():
        out_pt = pt_model(
            input_ids_pt,
            last_hidden_pt,
            layer_idx_input=layer_idx_in,
            layer_idx_attn=layer_idx_attn,
            layer_idx_mlp=layer_idx_mlp,
            layer_idx_out=layer_idx_out,
            labels=labels_pt
        )
    pt_logits = out_pt["logits"].detach().cpu().numpy()
    pt_loss_v = out_pt["loss"].detach().cpu().numpy() if out_pt["loss"] is not None else None

    # 2) layer_idx torch.Tensor => jnp
    def to_jnp_idx(x):
        if isinstance(x, torch.Tensor):
            return jnp.array(x.detach().cpu().numpy(), dtype=jnp.int32)
        return x

    layer_idx_in_jnp   = to_jnp_idx(layer_idx_in)
    layer_idx_attn_jnp = to_jnp_idx(layer_idx_attn)
    layer_idx_mlp_jnp  = to_jnp_idx(layer_idx_mlp)
    layer_idx_out_jnp  = to_jnp_idx(layer_idx_out)

    # 3) Flax forward
    input_ids_fl   = jnp.array(input_ids_pt.detach().cpu().numpy(), dtype=jnp.int32)
    last_hidden_fl = jnp.array(last_hidden_pt.detach().cpu().numpy(), dtype=jnp.float32)
    labels_fl      = None
    if labels_pt is not None:
        labels_fl = jnp.array(labels_pt.detach().cpu().numpy(), dtype=jnp.int32)

    out_fl = fl_model.apply(
        {"params": fl_params["params"]},
        input_ids_fl,
        last_hidden_fl,
        layer_idx_input=layer_idx_in_jnp,
        layer_idx_attn=layer_idx_attn_jnp,
        layer_idx_mlp=layer_idx_mlp_jnp,
        layer_idx_out=layer_idx_out_jnp,
        labels=labels_fl
    )
    fl_logits = np.array(out_fl.logits)
    fl_loss_v = np.array(out_fl.loss) if out_fl.loss is not None else None

    # 4) measure diff
    diff_logits = np.abs(pt_logits - fl_logits).mean()
    msg = f"{label} logits diff = {diff_logits:.6e}"
    if pt_loss_v is not None and fl_loss_v is not None:
        diff_loss = abs(pt_loss_v - fl_loss_v)
        msg += f", loss diff={diff_loss:.6e}, (pt={pt_loss_v:.6f}, fl={fl_loss_v:.6f})"
    print(msg)
    return diff_logits


# -----------------------------------------------------------
# 5) main
# -----------------------------------------------------------
def main():
    print("=== layer_unit_test_DepthDecoder.py (FIXED) ===")
    print("Covers 3 modes: (1) layer_idx=0, (2) layer_idx=None(S==num_codebooks), (3) 1D idx")

    config = DummyDepthConfig(
        vocab_size=13,
        audio_vocab_size=10,
        hidden_size=8,
        num_codebooks=3,
        input_size=8,
        num_hidden_layers=2
    )

    # 1) build PT, FL
    pt_model = MoshiDepthDecoderPT(config).eval()
    fl_model = MoshiDepthDecoderFL(config)

    # 2) sample input for param load
    B, S = 2, 4
    input_ids_pt = torch.randint(0, 2, (B, S))
    hidden_pt    = torch.randn(B, S, config.input_size)
    labels_pt    = torch.randint(0, config.audio_vocab_size, (B, S))

    # init (Flax)
    rng = jax.random.PRNGKey(0)
    fl_init = fl_model.init(
        rng,
        jnp.array(input_ids_pt.cpu().numpy(), dtype=jnp.int32),
        jnp.array(hidden_pt.cpu().numpy(),    dtype=jnp.float32),
        layer_idx_input=0,
        layer_idx_attn=0,
        layer_idx_mlp=0,
        layer_idx_out=0
    )
    # compare uninit
    compare_depth_decoder_pt_fl(
        pt_model, fl_model, fl_init,
        input_ids_pt, hidden_pt,
        layer_idx_in=0, layer_idx_attn=0, layer_idx_mlp=0, layer_idx_out=0,
        labels_pt=labels_pt,
        label="(Unloaded: layer_idx=0)"
    )

    # load PT->FL
    rng2 = jax.random.PRNGKey(999)
    fl_loaded = load_depth_decoder_pt_to_flax(
        pt_model,
        fl_model,
        rng2,
        input_ids_pt,
        hidden_pt
    )
    compare_depth_decoder_pt_fl(
        pt_model, fl_model, fl_loaded,
        input_ids_pt, hidden_pt,
        layer_idx_in=0, layer_idx_attn=0, layer_idx_mlp=0, layer_idx_out=0,
        labels_pt=labels_pt,
        label="(Loaded: layer_idx=0)"
    )

    # ----------------------------------------------------------------
    # (2) layer_idx=None & S==num_codebooks => let's do that
    # ----------------------------------------------------------------
    print("\n** S==num_codebooks => layer_idx=None test **")
    S2 = config.num_codebooks  # => e.g. 3
    input_ids2 = torch.randint(0, 2, (B, S2))
    hidden2    = torch.randn(B, S2, config.input_size)
    labels2    = torch.randint(0, config.audio_vocab_size, (B, S2))

    compare_depth_decoder_pt_fl(
        pt_model, fl_model, fl_loaded,
        input_ids2, hidden2,
        layer_idx_in=None,
        layer_idx_attn=None,
        layer_idx_mlp=None,
        layer_idx_out=None,
        labels_pt=labels2,
        label="(2) S==num_codebooks, layer_idx=None"
    )

    # ----------------------------------------------------------------
    # (3) tokenwise codebook => 1D idx
    # ----------------------------------------------------------------
    print("\n** tokenwise 1D idx usage **")
    S3 = 5
    input_ids3 = torch.randint(0, 2, (B, S3))
    hidden3    = torch.randn(B, S3, config.input_size)
    labels3    = torch.randint(0, config.audio_vocab_size, (B, S3))

    # ex) shape(S3,) => [0,1,2,1,0]
    layer_idx_1d = torch.tensor([0,1,2,1,0], dtype=torch.long)

    compare_depth_decoder_pt_fl(
        pt_model, fl_model, fl_loaded,
        input_ids3, hidden3,
        layer_idx_in=layer_idx_1d,
        layer_idx_attn=layer_idx_1d,
        layer_idx_mlp=layer_idx_1d,
        layer_idx_out=layer_idx_1d,
        labels_pt=labels3,
        label="(3) tokenwise 1D idx usage"
    )

    print("=== Done ===")


if __name__=="__main__":
    main()
