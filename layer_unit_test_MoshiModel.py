#!/usr/bin/env python3
# coding=utf-8

"""
layer_unit_test_MoshiModel.py

- 목적: Flexible Linear on/off, codebook 개수 변화를 다양하게 테스트.
- 크게 3가지 케이스를 각각 시험:
  (A) use_flexible_linear=False (baseline)
  (B) use_flexible_linear=True, num_codebooks=1
  (C) use_flexible_linear=True, num_codebooks=2

각 케이스에서:
1) PyTorch MoshiModelPT vs Flax MoshiModelFL 구성
2) Unloaded (무작위 초기화) 상태에서 비교
3) PyTorch -> Flax 파라미터 로딩 후 비교
4) 최종적으로 mean diff가 작으면 성공

추가:
- "S != num_layers in PT flexible usage" 에러를 피하기 위해, PyTorch 모델 쪽도
  use_flexible_linear=True 면 layer_idx=0로 통일하여 토큰 길이와 상관없이 단일 codebook을 사용한다.

더 나아가:
(4) "토큰별 다른 codebook 인덱스" 시나리오를 테스트 (codebooks>=2인 경우).
- 시퀀스 길이(S) 각 토큰마다 codebook idx를 다르게 준다. (shape=(S,) 인덱스 텐서)
- PyTorch, Flax에서 각각 forward를 manual 호출 후 결과 diff 확인.
- Flax 측은 '모델 전체 apply'에 `codebook_idx`를 넣어야 하므로, MoshiModelFL.__call__을 확장.
"""

import copy
import math
import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn

import torch
import torch.nn as pt_nn
import torch.nn.functional as F

from layer_unit_test_TPU_compare3 import (
    MoshiRMSNormPT, MoshiRMSNormFL,
    MoshiGatingMLPPT, MoshiGatingMLPFL,
    MoshiFlexibleLinearPT, MoshiFlexibleLinearFL,
    load_pt_weights_flexlinear, load_pt_weights_rmsnorm, load_pt_weights_gatingmlp,
)
from layer_unit_test_TPU_att_rope_compare3 import (
    MoshiAttentionPT,
    MoshiAttentionFL,
    load_pytorch_weights_into_flax_attention,
)
from layer_unit_test_MoshiDecoderLayer import (
    MoshiDecoderLayerPT,
    MoshiDecoderLayerFL,
    load_decoder_layer_pt_to_flax,
    compare_decoder_layers_pt_fl,
)

################################################################################
# 수정: MoshiDecoderLayerFL 내에서 flexible 사용 시 layer_idx=0 강제
################################################################################
# layer_unit_test_MoshiDecoderLayer.py에 있는 MoshiDecoderLayerFL를 여기서 다시 정의
# (동일 이름)
class MoshiDecoderLayerFL(nn.Module):
    config: any
    use_flexible_linear: bool = False
    use_rope: bool = True

    def setup(self):
        self.input_layernorm = MoshiRMSNormFL(
            self.config.hidden_size, eps=self.config.rms_norm_eps, name="input_layernorm"
        )
        self.self_attn = MoshiAttentionFL(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            head_dim=self.config.head_dim,
            num_codebooks=(self.config.num_codebooks if self.use_flexible_linear else 1),
            use_flexible_linear=self.use_flexible_linear,
            rope=self.use_rope,
            rope_theta=getattr(self.config, "rope_theta", 10000.0),
            rope_type="default",
            name="self_attn",
        )
        self.post_attention_layernorm = MoshiRMSNormFL(
            self.config.hidden_size, eps=self.config.rms_norm_eps, name="post_attention_layernorm"
        )
        self.mlp = MoshiGatingMLPFL(
            hidden_size=self.config.hidden_size,
            ffn_dim=self.config.ffn_dim,
            num_codebooks=(self.config.num_codebooks if self.use_flexible_linear else 1),
            hidden_act=self.config.hidden_act,
            name="mlp",
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray = None,
        position_ids: jnp.ndarray = None,
        layer_idx_attn=None,
        layer_idx_mlp=None
    ):
        if self.use_flexible_linear:
            if layer_idx_attn is None:
                layer_idx_attn = 0
            if layer_idx_mlp is None:
                layer_idx_mlp = 0

        x = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            x, attention_mask=attention_mask, position_ids=position_ids, layer_idx=layer_idx_attn
        )
        x = hidden_states + attn_out

        x2 = self.post_attention_layernorm(x)
        x2 = self.mlp(x2, layer_idx=layer_idx_mlp)
        return x + x2


################################################################################
# Config
################################################################################
class DummyMoshiConfig:
    def __init__(
        self,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=4,
        vocab_size=64,
        max_position_embeddings=32,
        rope_theta=10000.0,
        rope_scaling=None,
        use_flexible_linear=False,
        num_codebooks=1,
        ffn_dim=32,
        hidden_act="gelu",
        rms_norm_eps=1e-6,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_flexible_linear = use_flexible_linear
        self.num_codebooks = num_codebooks
        self.ffn_dim = ffn_dim
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps


################################################################################
# PyTorch model
################################################################################
class MoshiDecoderLayerPT_MINE(pt_nn.Module):
    def __init__(self, config: DummyMoshiConfig, layer_idx: int, use_rope=True):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = MoshiRMSNormPT(config.hidden_size, eps=config.rms_norm_eps)

        self.self_attn = MoshiAttentionPT(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_codebooks=(config.num_codebooks if config.use_flexible_linear else 1),
            use_flexible_linear=config.use_flexible_linear,
            rope=use_rope,
            rope_theta=config.rope_theta,
            rope_type="default",
        )
        self.post_attention_layernorm = MoshiRMSNormPT(config.hidden_size, eps=config.rms_norm_eps)

        if config.use_flexible_linear:
            self.mlp = MoshiGatingMLPPT(
                hidden_size=config.hidden_size,
                ffn_dim=config.ffn_dim,
                num_codebooks=config.num_codebooks,
                hidden_act=config.hidden_act,
            )
        else:
            self.mlp = MoshiGatingMLPPT(
                hidden_size=config.hidden_size,
                ffn_dim=config.ffn_dim,
                num_codebooks=1,
                hidden_act=config.hidden_act,
            )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, layer_idx_attn=None, layer_idx_mlp=None):
        if layer_idx_attn is None and self.config.use_flexible_linear:
            layer_idx_attn = 0
        if layer_idx_mlp is None and self.config.use_flexible_linear:
            layer_idx_mlp = 0

        x = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            x, attention_mask=attention_mask, position_ids=position_ids, layer_idx=layer_idx_attn
        )
        x = hidden_states + attn_out

        x2 = self.post_attention_layernorm(x)
        x2 = self.mlp(x2, layer_idx=layer_idx_mlp)
        return x + x2


class MoshiModelPT(pt_nn.Module):
    def __init__(self, config: DummyMoshiConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = pt_nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = pt_nn.ModuleList(
            [
                MoshiDecoderLayerPT_MINE(config, i, use_rope=True)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = MoshiRMSNormPT(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, position_ids=None, layer_idx_attn=None, layer_idx_mlp=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask, position_ids, layer_idx_attn, layer_idx_mlp)
        x = self.norm(x)
        return x


################################################################################
# Flax model
################################################################################
class MoshiEmbedFL(nn.Module):
    vocab_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, input_ids):
        emb = self.param("embedding", nn.initializers.normal(stddev=1.0), (self.vocab_size, self.hidden_size))
        return emb[input_ids, :]


class MoshiModelFL(nn.Module):
    config: DummyMoshiConfig

    def setup(self):
        self.embed_tokens = MoshiEmbedFL(self.config.vocab_size, self.config.hidden_size, name="embed_tokens")
        self.layers = [
            MoshiDecoderLayerFL(
                config=self.config,
                use_flexible_linear=self.config.use_flexible_linear,
                use_rope=True,
                name=f"layer_{i}"
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.final_norm = MoshiRMSNormFL(self.config.hidden_size, eps=self.config.rms_norm_eps, name="norm")

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray = None,
        position_ids: jnp.ndarray = None,
        layer_idx_attn=None,
        layer_idx_mlp=None
    ):
        x = self.embed_tokens(input_ids)
        for lyr in self.layers:
            x = lyr(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_idx_attn=layer_idx_attn,
                layer_idx_mlp=layer_idx_mlp
            )
        x = self.final_norm(x)
        return x


################################################################################
# Load / compare
################################################################################
def merge_params_dict(base_params, update_dict):
    newp = copy.deepcopy(base_params)
    for k,v in update_dict.items():
        if isinstance(v, dict) and k in newp:
            newp[k] = merge_params_dict(newp[k], v)
        else:
            newp[k] = v
    return newp

def load_moshi_model_pt_to_flax(pt_model, fl_model, rng, sample_input_ids, attention_mask=None, position_ids=None):
    B,S = sample_input_ids.shape
    init_vars = fl_model.init(
        rng,
        jnp.array(sample_input_ids.detach().cpu().numpy(), dtype=jnp.int32),
        None if attention_mask is None else jnp.array(attention_mask.detach().cpu().numpy()),
        None if position_ids is None else jnp.array(position_ids.detach().cpu().numpy())
    )
    params = copy.deepcopy(init_vars["params"])

    with torch.no_grad():
        emb_w = pt_model.embed_tokens.weight.detach().cpu().numpy()
    params["embed_tokens"]["embedding"] = jnp.array(emb_w)

    from layer_unit_test_TPU_att_rope_compare3 import load_pytorch_weights_into_flax_attention
    from layer_unit_test_TPU_compare3 import load_pt_weights_gatingmlp

    for i, pt_layer in enumerate(pt_model.layers):
        fl_layer_dict = params[f"layer_{i}"]

        fl_layer_dict["input_layernorm"]["weight"] = jnp.array(
            pt_layer.input_layernorm.weight.detach().cpu().numpy()
        )

        attn_fl_dummy = MoshiAttentionFL(
            hidden_size=pt_model.config.hidden_size,
            num_heads=pt_model.config.num_attention_heads,
            head_dim=pt_model.config.head_dim,
            num_codebooks=pt_model.config.num_codebooks,
            use_flexible_linear=pt_model.config.use_flexible_linear,
            rope=True,
            rope_theta=pt_model.config.rope_theta,
            rope_type="default"
        )
        dummy_hidden = torch.randn(2,3,pt_model.config.hidden_size)
        attn_vars = load_pytorch_weights_into_flax_attention(
            pt_layer.self_attn, attn_fl_dummy, jax.random.PRNGKey(999), dummy_hidden
        )
        fl_layer_dict["self_attn"] = merge_params_dict(fl_layer_dict["self_attn"], attn_vars["params"])

        fl_layer_dict["post_attention_layernorm"]["weight"] = jnp.array(
            pt_layer.post_attention_layernorm.weight.detach().cpu().numpy()
        )

        fl_mlp_dummy = MoshiGatingMLPFL(
            hidden_size=pt_layer.config.hidden_size,
            ffn_dim=pt_layer.config.ffn_dim,
            num_codebooks=(pt_layer.config.num_codebooks if pt_layer.config.use_flexible_linear else 1),
            hidden_act=pt_layer.config.hidden_act
        )
        dummy_x = torch.randn(2,4,pt_layer.config.hidden_size)
        mlp_vars = load_pt_weights_gatingmlp(pt_layer.mlp, fl_mlp_dummy, jax.random.PRNGKey(1000), dummy_x, layer_idx=0)
        fl_layer_dict["mlp"] = merge_params_dict(fl_layer_dict["mlp"], mlp_vars["params"])

        params[f"layer_{i}"] = fl_layer_dict

    params["norm"]["weight"] = jnp.array(pt_model.norm.weight.detach().cpu().numpy())
    final_vars = copy.deepcopy(init_vars)
    final_vars["params"] = params
    return final_vars

def compare_moshi_model(pt_model, fl_model, fl_params, input_ids, attention_mask=None, position_ids=None, label=""):
    with torch.no_grad():
        out_pt = pt_model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    pt_out_np = out_pt.detach().cpu().numpy()

    am_jnp = None
    pi_jnp = None
    if attention_mask is not None:
        am_jnp = jnp.array(attention_mask.detach().cpu().numpy())
    if position_ids is not None:
        pi_jnp = jnp.array(position_ids.detach().cpu().numpy())

    out_fl = fl_model.apply(
        {"params": fl_params["params"]},
        jnp.array(input_ids.detach().cpu().numpy()),
        am_jnp,
        pi_jnp,
    )
    fl_out_np = np.array(out_fl)

    diff = np.mean(np.abs(pt_out_np - fl_out_np))
    print(f"{label} mean diff = {diff:.6e}")
    return diff

################################################################################
# Tokenwise codebook usage test
################################################################################
def tokenwise_forward_pt(pt_model, input_ids, codebook_idx, attention_mask=None, position_ids=None):
    with torch.no_grad():
        return pt_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_idx_attn=codebook_idx,  # shape=(S,)
            layer_idx_mlp=codebook_idx
        )

def tokenwise_forward_fl(fl_model, fl_params, input_ids, codebook_idx, attention_mask=None, position_ids=None):
    am_jnp = None
    pi_jnp = None
    if attention_mask is not None:
        am_jnp = jnp.array(attention_mask.detach().cpu().numpy())
    if position_ids is not None:
        pi_jnp = jnp.array(position_ids.detach().cpu().numpy())

    input_jnp = jnp.array(input_ids.detach().cpu().numpy(), dtype=jnp.int32)
    codebook_jnp = jnp.array(codebook_idx.detach().cpu().numpy(), dtype=jnp.int32)

    out_fl = fl_model.apply(
        {"params": fl_params},
        input_jnp,
        am_jnp,
        pi_jnp,
        layer_idx_attn=codebook_jnp,
        layer_idx_mlp=codebook_jnp
    )
    return np.array(out_fl)

def test_tokenwise_codebook(pt_model, fl_model, fl_vars, input_ids, attention_mask, position_ids):
    B,S = input_ids.shape
    codebook_idx_pt = torch.randint(0, pt_model.config.num_codebooks, (S,), dtype=torch.long)
    print(f"  [Tokenwise codebook test] codebook_idx = {codebook_idx_pt.tolist()}")

    pt_out = tokenwise_forward_pt(pt_model, input_ids, codebook_idx_pt, attention_mask, position_ids)
    pt_out_np = pt_out.detach().cpu().numpy()

    fl_out_np = tokenwise_forward_fl(fl_model, fl_vars["params"], input_ids, codebook_idx_pt, attention_mask, position_ids)
    diff = np.mean(np.abs(pt_out_np - fl_out_np))
    print(f"  [Tokenwise codebook test] PT vs FL mean diff = {diff:.6e}")

################################################################################
# main
################################################################################
def main():
    print("=== Testing FlexibleLinear On/Off, codebook variations + tokenwise codebook usage ===")

    scenarios = [
        {"use_flexible_linear": False, "num_codebooks": 1},
        {"use_flexible_linear": True,  "num_codebooks": 1},
        {"use_flexible_linear": True,  "num_codebooks": 2},
    ]

    B,S = 2,5
    input_ids_pt = torch.randint(0,64,(B,S))
    attention_mask_pt = torch.zeros((B,1,S,S))
    for i in range(S):
        attention_mask_pt[:,:,i,i+1:] = float("-inf")
    position_ids_pt = torch.arange(S).unsqueeze(0).repeat(B,1)

    for idx, sc in enumerate(scenarios):
        config = DummyMoshiConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            head_dim=4,
            vocab_size=64,
            max_position_embeddings=32,
            rope_theta=10000.0,
            rope_scaling=None,
            use_flexible_linear=sc["use_flexible_linear"],
            num_codebooks=sc["num_codebooks"],
            ffn_dim=32,
            hidden_act="gelu",
            rms_norm_eps=1e-6,
        )
        label_s = f"(Scenario {idx+1}) flexible={sc['use_flexible_linear']}, codebooks={sc['num_codebooks']}"

        print("\n============================================")
        print(f"{label_s}: building PT/FL model")

        pt_model = MoshiModelPT(config).eval()
        fl_model = MoshiModelFL(config)

        init_vars = fl_model.init(
            jax.random.PRNGKey(0),
            jnp.array(input_ids_pt.detach().cpu().numpy(), dtype=jnp.int32),
            jnp.array(attention_mask_pt.detach().cpu().numpy(), dtype=jnp.float32),
            jnp.array(position_ids_pt.detach().cpu().numpy(), dtype=jnp.int32)
        )
        diff_before = compare_moshi_model(
            pt_model, fl_model, init_vars,
            input_ids_pt, attention_mask_pt, position_ids_pt,
            label=label_s+"(unloaded)"
        )

        fl_loaded = load_moshi_model_pt_to_flax(
            pt_model, fl_model,
            jax.random.PRNGKey(42),
            input_ids_pt, attention_mask_pt, position_ids_pt
        )
        diff_after = compare_moshi_model(
            pt_model, fl_model, fl_loaded,
            input_ids_pt, attention_mask_pt, position_ids_pt,
            label=label_s+"(loaded)"
        )
        print(f"{label_s} => diff_before={diff_before:.6e}, diff_after={diff_after:.6e}")

        if sc["use_flexible_linear"] and sc["num_codebooks"]>1:
            print(f"{label_s}: Try tokenwise codebook usage test:")
            test_tokenwise_codebook(
                pt_model,
                fl_model,
                fl_loaded,
                input_ids_pt,
                attention_mask_pt,
                position_ids_pt
            )

if __name__=="__main__":
    main()
