#!/usr/bin/env python3
# coding=utf-8

"""
layer_unit_test_TPU_att_rope_compare3.py

- 3가지 버전의 MoshiAttention 비교:
  (A) *PT_origin  : modeling_moshi.py의 원본 PyTorch 코드 복사
  (B) *PT         : "layer_unit_test_TPU_att_rope.py"의 PyTorch 코드(간소화 X, 원본 그대로)
  (C) *FL         : JAX/Flax 코드

- use_flexible_linear=False / True 각각 비교
- RoPE, flexible linear, etc. 모두 테스트
- 최종적으로 *PT_origin vs. *FL 일치가 가장 중요, 추가로 *PT vs. *FL, *PT_origin vs. *PT도 함께 확인.
"""
from typing import Any, Optional, Union

import math
import numpy as np

# --------------------
# JAX / Flax
# --------------------
import jax
import jax.numpy as jnp
from flax import linen as nn

# --------------------
# PyTorch
# --------------------
import torch
import torch.nn as pt_nn
import torch.nn.functional as F


###############################################################################
# 0) 공용 config & rope init
###############################################################################
def rope_init_default(config, device=None, seq_len=None):
    hidden_dim = config.hidden_size // config.num_attention_heads
    inv_freq = torch.tensor(
        1.0 / (10000.0 ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim)),
        device=device
    )
    attention_scaling = 1.0
    return inv_freq, attention_scaling

def rope_init_dynamic(config, device=None, seq_len=None):
    hidden_dim = config.hidden_size // config.num_attention_heads
    if seq_len is None:
        seq_len = config.max_position_embeddings
    inv_freq = torch.tensor(
        1.0 / (config.rope_theta ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim)),
        device=device
    )
    attention_scaling = 1.0 / (seq_len ** 0.05)
    return inv_freq, attention_scaling

ROPE_INIT_FUNCTIONS = {
    "default": rope_init_default,
    "dynamic": rope_init_dynamic,
}

class DummyMoshiConfig:
    def __init__(
        self,
        hidden_size=16,
        num_attention_heads=4,
        rope_scaling=None,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        num_codebooks=1,
        head_dim=4
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_codebooks = num_codebooks
        self.head_dim = head_dim


###############################################################################
# 1) [원본] PT classes from modeling_moshi.py => *_PT_origin
###############################################################################
# -- (A) PT_origin: 그대로 유지 --
class MoshiFlexibleLinearPT_origin(pt_nn.Module):
    """
    원본 PyTorch flexible linear from modeling_moshi.py
    (기존 'MoshiFlexibleLinear' 이름, 여기선 suffix _PT_origin)
    """
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        # Stack weights: (num_layers, output_size, input_size)
        self.weight = pt_nn.Parameter(torch.randn(num_layers, output_size, input_size))

    def forward(self, x, layer_idx=None):
        if x.dim()==2:
            x = x.unsqueeze(1)
        B, S, D = x.shape
        nl, out_dim, in_dim = self.weight.shape
        if D != in_dim:
            raise ValueError("dim mismatch in MoshiFlexibleLinearPT_origin")

        if layer_idx is not None:
            if isinstance(layer_idx, int):
                w = self.weight[layer_idx]
                out = F.linear(x, w)  # shape (B,S,out_dim)
                return out
            elif layer_idx.dim()==1:
                L = layer_idx.shape[0]
                if S == L:
                    outs = []
                    for i in range(L):
                        outs.append(F.linear(x[:,i], self.weight[layer_idx[i]]))
                    return torch.stack(outs, dim=1)
                if S==1:
                    x_squeezed = x[:,0]
                    outs = []
                    for idx in layer_idx:
                        outs.append(F.linear(x_squeezed, self.weight[idx]))
                    return torch.stack(outs, dim=1)
                else:
                    raise ValueError("shape mismatch for PT_origin flexible linear usage")
            else:
                raise ValueError("layer_idx must be int or 1D tensor for PT_origin usage")
        else:
            if S!=nl:
                raise ValueError("S!=num_layers in PT_origin flexible linear usage")
            outs = []
            for i in range(S):
                w_i = self.weight[i]
                outs.append(F.linear(x[:,i], w_i))
            return torch.stack(outs, dim=1)

class MoshiLinearPT_origin(pt_nn.Module):
    def __init__(self, input_dim, output_dim, num_codebooks, use_flexible_linear=False):
        super().__init__()
        self.use_flexible_linear = use_flexible_linear
        if not use_flexible_linear:
            self.linear = pt_nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = MoshiFlexibleLinearPT_origin(input_dim, output_dim, num_codebooks)

    def forward(self, x, layer_idx=None):
        if self.use_flexible_linear:
            return self.linear(x, layer_idx=layer_idx)
        else:
            return self.linear(x)

def rotate_half_pt_origin(x):
    x1 = x[..., : x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb_pt_origin(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_emb = (q * cos) + (rotate_half_pt_origin(q) * sin)
    k_emb = (k * cos) + (rotate_half_pt_origin(k) * sin)
    return q_emb, k_emb

def repeat_kv_pt_origin(hidden_states: torch.Tensor, n_rep: int):
    batch, num_kv, slen, head_dim = hidden_states.shape
    if n_rep==1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv*n_rep, slen, head_dim)

class MoshiRotaryEmbeddingPT_origin(pt_nn.Module):
    def __init__(self, config: DummyMoshiConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type","default")
        else:
            self.rope_type = "default"
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids)+1
        if seq_len>self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        if seq_len< self.original_max_seq_len and self.max_seq_len_cached>self.original_max_seq_len:
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, x.device)

        bsz, seq_len = position_ids.shape
        p_float = position_ids.float()
        inv_f = self.inv_freq.float()
        hidden_dim = self.config.hidden_size // self.config.num_attention_heads

        freqs = torch.einsum("i,d->id", p_float[0], inv_f)
        cos_half = freqs.cos() * self.attention_scaling
        sin_half = freqs.sin() * self.attention_scaling
        cos = torch.cat([cos_half, cos_half], dim=-1)
        sin = torch.cat([sin_half, sin_half], dim=-1)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        return cos, sin

class MoshiAttentionPT_origin(pt_nn.Module):
    def __init__(
        self,
        config: DummyMoshiConfig,
        use_rope=True,
        use_flexible_linear=False
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_attention_heads
        self.num_key_value_groups = self.num_heads//self.num_key_value_heads
        self.scaling = 1./math.sqrt(self.head_dim)
        self.use_rope = use_rope

        self.q_proj = MoshiLinearPT_origin(
            config.hidden_size, self.num_heads*self.head_dim,
            config.num_codebooks, use_flexible_linear
        )
        self.k_proj = MoshiLinearPT_origin(
            config.hidden_size, self.num_key_value_heads*self.head_dim,
            config.num_codebooks, use_flexible_linear
        )
        self.v_proj = MoshiLinearPT_origin(
            config.hidden_size, self.num_key_value_heads*self.head_dim,
            config.num_codebooks, use_flexible_linear
        )
        self.o_proj = MoshiLinearPT_origin(
            self.num_heads*self.head_dim, config.hidden_size,
            config.num_codebooks, use_flexible_linear
        )

        self.rotary_emb = None
        if use_rope:
            self.rotary_emb = MoshiRotaryEmbeddingPT_origin(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, seq_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

        if self.rotary_emb is not None and position_ids is not None:
            cos,sin = self.rotary_emb(v, position_ids)
            q,k = apply_rotary_pos_emb_pt_origin(q,k, cos, sin)

        k = repeat_kv_pt_origin(k, self.num_key_value_groups)
        v = repeat_kv_pt_origin(v, self.num_key_value_groups)

        attn_weights = torch.matmul(q, k.transpose(-2,-1))*self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


###############################################################################
# 2) [테스트/간소화] PT classes => *_PT
#    --> 여기 부분만 "layer_unit_test_TPU_att_rope.py" 내용 그대로 반영 (간소화 X)
###############################################################################
# -- (B) *PT: 아래는 "layer_unit_test_TPU_att_rope.py"에서 가져온 PyTorch 코드 (간소화 없음) --

# 1) FlexibleLinear
class MoshiFlexibleLinearPT(pt_nn.Module):
    """
    Flexible linear (PyTorch). Stacks multiple (out_dim,in_dim) in weight.
    shape: (num_layers, output_size, input_size)
    """
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        self.num_layers = num_layers
        self.weight = pt_nn.Parameter(torch.randn(num_layers, output_size, input_size))

    def forward(self, x, layer_idx=None):
        """
        x: (B, S, in_dim) or (B, in_dim)
        layer_idx: None / int / 1D tensor
        - None => S==num_layers
        - int => single layer for all
        - 1D => shape match
        """
        if x.dim() == 2:
            # (B,in_dim) -> (B,1,in_dim)
            x = x.unsqueeze(1)

        B, S, D = x.shape
        nl, out_dim, in_dim = self.weight.shape
        if D != in_dim:
            raise ValueError("Dim mismatch in MoshiFlexibleLinearPT")

        if layer_idx is None:
            if S != nl:
                raise ValueError("S != num_layers for flexible usage")
            # each i -> weight[i]
            outs = []
            for i in range(S):
                w_i = self.weight[i]
                outs.append(F.linear(x[:, i], w_i))
            return torch.stack(outs, dim=1)

        if isinstance(layer_idx, int):
            # single weight
            if not (0 <= layer_idx < nl):
                raise ValueError("invalid layer_idx in PT flexible linear usage.")
            w = self.weight[layer_idx]
            return F.linear(x, w)

        # layer_idx is 1D
        if layer_idx is not None and layer_idx.dim() == 1:
            L = layer_idx.shape[0]
            if S == L:
                outs = []
                for i in range(L):
                    outs.append(F.linear(x[:, i], self.weight[layer_idx[i]]))
                return torch.stack(outs, dim=1)
            if S == 1:
                x_squeezed = x[:, 0]
                outs = []
                for idx in layer_idx:
                    outs.append(F.linear(x_squeezed, self.weight[idx]))
                return torch.stack(outs, dim=1)
            raise ValueError("shape mismatch for layer_idx (PT)")

        raise ValueError("layer_idx must be None/int/1D in PT flexible linear usage.")


# 2) Normal / Flexible linear wrapper
class MoshiLinearPT(pt_nn.Module):
    """
    Wrapper that either uses standard Linear or MoshiFlexibleLinearPT
    depending on use_flexible_linear & num_codebooks.
    """
    def __init__(self, input_dim, output_dim, num_codebooks, use_flexible_linear=False):
        super().__init__()
        self.use_flexible_linear = use_flexible_linear
        if not use_flexible_linear:
            # normal
            self.linear = pt_nn.Linear(input_dim, output_dim, bias=False)
        else:
            # flexible
            self.linear = MoshiFlexibleLinearPT(input_dim, output_dim, num_codebooks)

    def forward(self, x, layer_idx=None):
        if not self.use_flexible_linear:
            return self.linear(x)
        else:
            return self.linear(x, layer_idx=layer_idx)


# 3) rotate_half, apply_rotary_pos_emb, repeat_kv
def rotate_half(x):
    """
    x shape: (..., head_dim).
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    q,k: (B,nHeads,S,head_dim).
    cos,sin: (1,S,head_dim) => unsqueeze => (1,1,S,head_dim).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
    """
    Repeats K,V if needed. shape: (B,nHeads,S,head_dim).
    If n_rep=1 => no effect
    """
    b, n, s, h = hidden_states.shape
    if n_rep==1:
        return hidden_states
    hs = hidden_states[:, :, None, :, :].expand(b, n, n_rep, s, h)
    return hs.reshape(b, n*n_rep, s, h)


# 4) MoshiRotaryEmbeddingPT
class MoshiRotaryEmbeddingPT(pt_nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if getattr(config, "rope_scaling", None):
            self.rope_type = config.rope_scaling.get("rope_type","default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq.clone()

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        if seq_len < self.original_max_seq_len and self.max_seq_len_cached>self.original_max_seq_len:
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        bsz, s_len = position_ids.shape
        p_float = position_ids.float()
        inv_f = self.inv_freq.float()
        hidden_dim = self.config.hidden_size // self.config.num_attention_heads

        freqs = torch.einsum("i,d->id", p_float[0], inv_f)
        cos_half = freqs.cos() * self.attention_scaling
        sin_half = freqs.sin() * self.attention_scaling

        cos = torch.cat([cos_half, cos_half], dim=-1)
        sin = torch.cat([sin_half, sin_half], dim=-1)

        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        return cos, sin

'''
# 5) MoshiAttentionPT
class MoshiAttentionPT(pt_nn.Module):
    """
    PyTorch MoshiAttention, can toggle rope usage,
    can also toggle flexible linear usage in q_proj/k_proj/v_proj/o_proj.
    Now supports layer_idx to handle flexible usage for each projection.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim,
        num_codebooks=1,
        use_flexible_linear=False,
        rope=True,
        rope_theta=10000.0,
        rope_type="default"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_heads
        self.num_key_value_groups = 1
        self.scaling = 1.0 / (head_dim**0.5)
        self.use_rope = rope
        self.rope_type = rope_type

        config = DummyMoshiConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            rope_scaling={"rope_type": rope_type} if rope_type != "default" else None,
            rope_theta=rope_theta,
            num_codebooks=num_codebooks,
            head_dim=head_dim,
        )
        self.q_proj = MoshiLinearPT(hidden_size, num_heads*head_dim, num_codebooks, use_flexible_linear)
        self.k_proj = MoshiLinearPT(hidden_size, num_heads*head_dim, num_codebooks, use_flexible_linear)
        self.v_proj = MoshiLinearPT(hidden_size, num_heads*head_dim, num_codebooks, use_flexible_linear)
        self.o_proj = MoshiLinearPT(num_heads*head_dim, hidden_size, num_codebooks, use_flexible_linear)

        if rope:
            self.rotary_emb = MoshiRotaryEmbeddingPT(config)
        else:
            self.rotary_emb = None

        self.attention_dropout = 0.0

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        layer_idx=None   # <--- New argument
    ):
        """
        layer_idx: None or int or 1D tensor
          - if self.use_flexible_linear=True, we pass layer_idx to each projection (q/k/v/o).
        """
        bsz, q_len, _ = hidden_states.size()

        # Q, K, V
        q = self.q_proj(hidden_states, layer_idx=layer_idx)
        k = self.k_proj(hidden_states, layer_idx=layer_idx)
        v = self.v_proj(hidden_states, layer_idx=layer_idx)

        # reshape => (B, num_heads, seq_len, head_dim)
        q = q.view(bsz,q_len,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(bsz,q_len,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(bsz,q_len,self.num_heads,self.head_dim).transpose(1,2)

        # optional rope
        if self.use_rope and self.rotary_emb is not None and position_ids is not None:
            cos,sin = self.rotary_emb(v, position_ids)
            q,k = apply_rotary_pos_emb(q, k, cos, sin)

        # repeat kv if needed
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # scaled dot-product
        attn_weights = torch.matmul(q, k.transpose(2,3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # apply to v
        attn_output = torch.matmul(attn_weights, v).transpose(1,2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        # final projection
        attn_output = self.o_proj(attn_output, layer_idx=layer_idx)

        return attn_output
'''

class MoshiAttentionPT(pt_nn.Module):
    """
    PyTorch MoshiAttention, can toggle rope usage and flexible linear usage.
    Now supports `layer_idx` so we can do flexible usage in Q/K/V/O projections
    (if use_flexible_linear=True).
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_codebooks: int = 1,
        use_flexible_linear: bool = False,
        rope: bool = True,
        rope_theta: float = 10000.0,
        rope_type: str = "default"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_heads
        self.num_key_value_groups = 1
        self.scaling = 1.0 / math.sqrt(head_dim)
        self.use_rope = rope
        self.rope_type = rope_type

        # 예: DummyMoshiConfig는 사용자 프로젝트에 이미 존재한다고 가정
        config = DummyMoshiConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            rope_scaling={"rope_type": rope_type} if rope_type != "default" else None,
            rope_theta=rope_theta,
            num_codebooks=num_codebooks,
            head_dim=head_dim,
        )

        # Q/K/V/O projection
        # (MoshiLinearPT 내부에서 use_flexible_linear=True이면 MoshiFlexibleLinearPT가 쓰임)
        self.q_proj = MoshiLinearPT(hidden_size, num_heads * head_dim, num_codebooks, use_flexible_linear)
        self.k_proj = MoshiLinearPT(hidden_size, num_heads * head_dim, num_codebooks, use_flexible_linear)
        self.v_proj = MoshiLinearPT(hidden_size, num_heads * head_dim, num_codebooks, use_flexible_linear)
        self.o_proj = MoshiLinearPT(num_heads * head_dim, hidden_size, num_codebooks, use_flexible_linear)

        # Rotary embedding (RoPE) optional
        if rope:
            self.rotary_emb = MoshiRotaryEmbeddingPT(config)
        else:
            self.rotary_emb = None

        # optional dropout for attn weights
        self.attention_dropout = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_idx: Optional[Union[int, torch.Tensor]] = None,  # <--- 새로 추가
    ) -> torch.Tensor:
        """
        layer_idx: None/int/1D tensor.
          - If use_flexible_linear=True, we pass layer_idx to each projection
            so that different weight slices can be used per layer_idx.
          - If None, normal usage (no slicing).
        """

        bsz, seq_len, _ = hidden_states.size()

        # 1) Q, K, V proj (with layer_idx)
        q = self.q_proj(hidden_states, layer_idx=layer_idx)
        k = self.k_proj(hidden_states, layer_idx=layer_idx)
        v = self.v_proj(hidden_states, layer_idx=layer_idx)

        # 2) reshape => (B, num_heads, seq_len, head_dim)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) optional rotary embedding
        if self.use_rope and self.rotary_emb is not None and position_ids is not None:
            cos, sin = self.rotary_emb(v, position_ids)  # commonly pass 'v' just for device shape
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4) scaled dot-product
        # repeat_kv => (B, nHeads, S, head_dim) if we needed multiple groups
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # 5) apply attn weights to V
        attn_output = torch.matmul(attn_weights, v)  # (B, nHeads, S, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, -1)

        # 6) final output projection
        attn_output = self.o_proj(attn_output, layer_idx=layer_idx)

        return attn_output

###############################################################################
# 3) Flax classes => *_FL
###############################################################################
# -- (C) FL: 그대로 유지 --

class MoshiFlexibleLinearFL(nn.Module):
    input_size: int
    output_size: int
    num_layers: int

    @nn.compact
    def __call__(self, x, layer_idx=None):
        weight = self.param(
            "weight",
            nn.initializers.normal(stddev=1.0),
            (self.num_layers, self.output_size, self.input_size)
        )
        if x.ndim==2:
            x = x[:, None, :]
        B,S,D = x.shape
        if D!=self.input_size:
            raise ValueError("dim mismatch in FL flexible linear")

        def matmul_2d(vec, w_):
            return jnp.einsum("bd,od->bo", vec, w_)

        if layer_idx is None:
            if S!=self.num_layers:
                raise ValueError("S!=num_layers in FL usage")
            outs=[]
            for i in range(S):
                outs.append(matmul_2d(x[:,i], weight[i]))
            return jnp.stack(outs, axis=1)
        if isinstance(layer_idx, int):
            if not (0 <= layer_idx < self.num_layers):
                raise ValueError("invalid layer_idx in flax flexible usage")
            w_ = weight[layer_idx]
            outs=[]
            for i in range(S):
                outs.append(matmul_2d(x[:,i], w_))
            return jnp.stack(outs, axis=1)
        if layer_idx is not None and layer_idx.ndim==1:
            L = layer_idx.shape[0]
            if S==L:
                outs=[]
                for i in range(L):
                    outs.append(matmul_2d(x[:,i], weight[layer_idx[i]]))
                return jnp.stack(outs, axis=1)
            if S==1:
                x_squeezed = x[:,0]
                outs=[]
                for idx in layer_idx:
                    outs.append(matmul_2d(x_squeezed, weight[idx]))
                return jnp.stack(outs, axis=1)
            raise ValueError("shape mismatch in FL usage")
        raise ValueError("invalid layer_idx in FL usage")

class MoshiLinearFL(nn.Module):
    input_dim: int
    output_dim: int
    num_codebooks: int
    use_flexible_linear: bool=False

    @nn.compact
    def __call__(self, x, layer_idx=None):
        if not self.use_flexible_linear:
            kernel = self.param(
                "kernel",
                nn.initializers.normal(stddev=1.0),
                (self.input_dim, self.output_dim)
            )
            if x.ndim==2:
                out = jnp.dot(x, kernel)
            else:
                out = jnp.einsum("bsd,do->bso", x, kernel)
            return out
        else:
            flex_lin = MoshiFlexibleLinearFL(
                self.input_dim,
                self.output_dim,
                self.num_codebooks,
                name="flex_lin"
            )
            return flex_lin(x, layer_idx=layer_idx)

def rotate_half_fl(x):
    half = x.shape[-1]//2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb_fl(q, k, cos, sin):
    cos = jnp.expand_dims(cos, 1)
    sin = jnp.expand_dims(sin, 1)
    q_emb = (q*cos) + (rotate_half_fl(q)*sin)
    k_emb = (k*cos) + (rotate_half_fl(k)*sin)
    return q_emb, k_emb

def rope_init_fl(config: DummyMoshiConfig, seq_len=None):
    rope_type = "default"
    if config.rope_scaling is not None:
        rope_type = config.rope_scaling.get("rope_type","default")
    hidden_dim = config.hidden_size//config.num_attention_heads

    if rope_type=="dynamic":
        if seq_len is None:
            seq_len = config.max_position_embeddings
        inv_freq = np.array(1./(config.rope_theta**(np.arange(0,hidden_dim,2)/hidden_dim)), dtype=np.float32)
        attention_scaling = 1./(seq_len**0.05)
    else:
        inv_freq = np.array(1./(config.rope_theta**(np.arange(0,hidden_dim,2)/hidden_dim)), dtype=np.float32)
        attention_scaling=1.
    return inv_freq, attention_scaling

class MoshiRotaryEmbeddingFL(nn.Module):
    config: DummyMoshiConfig
    @nn.compact
    def __call__(self, x, position_ids):
        seq_len = jnp.max(position_ids)+1
        inv_freq, scaling = rope_init_fl(self.config, seq_len=seq_len)
        p32 = position_ids.astype(jnp.float32)
        freqs = jnp.einsum("i,d->id", p32[0], inv_freq)

        cos_half = jnp.cos(freqs)*scaling
        sin_half = jnp.sin(freqs)*scaling
        cos = jnp.concatenate([cos_half, cos_half], axis=-1)
        sin = jnp.concatenate([sin_half, sin_half], axis=-1)
        cos = cos[None,:]
        sin = sin[None,:]
        return cos, sin

def repeat_kv_fl(x:jnp.ndarray, n_rep:int):
    B,N,S,H = x.shape
    if n_rep==1:
        return x
    tile = jnp.tile(x[:, :, None, :, :], (1,1,n_rep,1,1))
    return tile.reshape((B, N*n_rep, S, H))

class MoshiAttentionFL(nn.Module):
    hidden_size: int
    num_heads: int
    head_dim: int
    num_codebooks: int=1
    use_flexible_linear: bool=False
    rope: bool=True
    rope_theta: float=10000.0
    rope_type: str="default"

    @nn.compact
    #def __call__(self, hidden_states, attention_mask=None, position_ids=None):
    #def __call__(self, hidden_states, attention_mask=None, position_ids=None, layer_idx=None):
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        layer_idx: Optional[Union[int, jnp.ndarray, torch.Tensor]] = None
    ):

        if isinstance(layer_idx, torch.Tensor):
            layer_idx = jnp.array(layer_idx.detach().cpu().numpy(), dtype=jnp.int32)

        B,S,D = hidden_states.shape

        q_lin = MoshiLinearFL(self.hidden_size, self.num_heads*self.head_dim, self.num_codebooks, self.use_flexible_linear, name="q_proj")
        k_lin = MoshiLinearFL(self.hidden_size, self.num_heads*self.head_dim, self.num_codebooks, self.use_flexible_linear, name="k_proj")
        v_lin = MoshiLinearFL(self.hidden_size, self.num_heads*self.head_dim, self.num_codebooks, self.use_flexible_linear, name="v_proj")
        o_lin = MoshiLinearFL(self.num_heads*self.head_dim, self.hidden_size, self.num_codebooks, self.use_flexible_linear, name="o_proj")

        #q = q_lin(hidden_states).reshape((B,S,self.num_heads,self.head_dim)).transpose((0,2,1,3))
        #k = k_lin(hidden_states).reshape((B,S,self.num_heads,self.head_dim)).transpose((0,2,1,3))
        #v = v_lin(hidden_states).reshape((B,S,self.num_heads,self.head_dim)).transpose((0,2,1,3))
        q = q_lin(hidden_states, layer_idx=layer_idx).reshape((B,S,self.num_heads,self.head_dim)).transpose((0,2,1,3))
        k = k_lin(hidden_states, layer_idx=layer_idx).reshape((B,S,self.num_heads,self.head_dim)).transpose((0,2,1,3))
        v = v_lin(hidden_states, layer_idx=layer_idx).reshape((B,S,self.num_heads,self.head_dim)).transpose((0,2,1,3))


        if self.rope and position_ids is not None:
            config = DummyMoshiConfig(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_heads,
                rope_scaling={"rope_type":self.rope_type} if self.rope_type!="default" else None,
                rope_theta=self.rope_theta
            )
            cos,sin = MoshiRotaryEmbeddingFL(config, name="rotary_emb")(v, position_ids)
            q,k = apply_rotary_pos_emb_fl(q,k, cos,sin)

        scale = 1./ jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q,k)*scale
        if attention_mask is not None:
            attn_weights = attn_weights+attention_mask
        attn_weights = nn.softmax(attn_weights, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        out = out.transpose((0,2,1,3)).reshape((B,S,self.num_heads*self.head_dim))

        #out = o_lin(out)
        out = o_lin(out, layer_idx=layer_idx)
        return out


###############################################################################
# 4) PyTorch->Flax 가중치 로드 함수
###############################################################################
def load_pytorch_weights_into_flax_attention(pt_attn, flax_attn, rng, sample_input, attention_mask=None, position_ids=None):
    sample_input_jnp = jnp.array(sample_input.detach().cpu().numpy())
    am_jnp=None
    pi_jnp=None
    if attention_mask is not None:
        am_jnp = jnp.array(attention_mask.detach().cpu().numpy())
    if position_ids is not None:
        pi_jnp = jnp.array(position_ids.detach().cpu().numpy())

    variables = flax_attn.init(rng, sample_input_jnp, am_jnp, pi_jnp)
    new_params = dict(variables["params"])

    for proj_name in ["q_proj","k_proj","v_proj","o_proj"]:
        pt_layer = getattr(pt_attn, proj_name)
        fl_sub = new_params[proj_name]

        if not pt_layer.use_flexible_linear:
            pt_w = pt_layer.linear.weight.detach().cpu().numpy()
            fl_sub["kernel"] = jnp.array(pt_w.T)
        else:
            pt_w = pt_layer.linear.weight.detach().cpu().numpy()
            flex_lin_dict = dict(fl_sub["flex_lin"])
            flex_lin_dict["weight"] = jnp.array(pt_w)
            fl_sub["flex_lin"] = flex_lin_dict

        new_params[proj_name] = fl_sub

    return {"params": new_params}


###############################################################################
# 5) 세 모델 출력 비교 함수
###############################################################################
def compare_two_model_outputs(pt_modelA, pt_modelB, hidden_states_pt, attention_mask=None, position_ids=None, labelA="A", labelB="B"):
    with torch.no_grad():
        outA = pt_modelA(hidden_states_pt, attention_mask=attention_mask, position_ids=position_ids)
        outB = pt_modelB(hidden_states_pt, attention_mask=attention_mask, position_ids=position_ids)
    diff = (outA-outB).abs().mean().item()
    print(f"  {labelA} vs {labelB} mean diff = {diff:.6e}")
    return diff

def compare_pt_with_fl(pt_model, fl_model, fl_params, hidden_states_pt, attention_mask=None, position_ids=None, labelPT="", labelFL="FL"):
    with torch.no_grad():
        out_pt = pt_model(hidden_states_pt, attention_mask=attention_mask, position_ids=position_ids)
    out_pt_np = out_pt.detach().cpu().numpy()

    am_jnp=None
    pi_jnp=None
    if attention_mask is not None:
        am_jnp = jnp.array(attention_mask.detach().cpu().numpy())
    if position_ids is not None:
        pi_jnp = jnp.array(position_ids.detach().cpu().numpy())

    out_fl = fl_model.apply({"params": fl_params}, jnp.array(hidden_states_pt.detach().cpu().numpy()), am_jnp, pi_jnp)
    out_fl_np = np.array(out_fl)

    diff = np.abs(out_pt_np - out_fl_np).mean()
    print(f"  {labelPT} vs {labelFL} mean diff = {diff:.6e}")
    return diff


###############################################################################
# 6) Demo main
###############################################################################
def main():
    print("==== 3-way compare: (PT_origin), (PT_test), (FL) ====")

    # ----------------------------------------------------
    # scenario A) flexible=False, rope_type="dynamic"
    # ----------------------------------------------------
    print("\n--- Scenario A: flexible=False, rope_type=dynamic ---")

    configA = DummyMoshiConfig(hidden_size=16, num_attention_heads=4, head_dim=4,
                               rope_scaling={"rope_type":"dynamic"}, rope_theta=10000., num_codebooks=1)
    # 1) PT_origin
    pt_attn_originA = MoshiAttentionPT_origin(configA, use_rope=True, use_flexible_linear=False).eval()
    # 2) PT_test
    pt_attn_testA = MoshiAttentionPT(
        hidden_size=16,
        num_heads=4,
        head_dim=4,
        num_codebooks=1,
        use_flexible_linear=False,
        rope=True,
        rope_theta=10000.0,
        rope_type="dynamic"
    ).eval()
    # 3) FL
    fl_attnA = MoshiAttentionFL(
        hidden_size=16, num_heads=4, head_dim=4,
        num_codebooks=1, use_flexible_linear=False,
        rope=True, rope_theta=10000., rope_type="dynamic"
    )

    B,S = 2,7
    x_ptA = torch.randn(B,S,16)
    pos_idsA = torch.tensor([[0,1,2,3,10,11,14],[0,1,2,9,10,13,14]], dtype=torch.long)

    # PT_origin -> FL 가중치 로드
    rngA = jax.random.PRNGKey(0)
    fl_varsA = load_pytorch_weights_into_flax_attention(pt_attn_originA, fl_attnA, rngA, x_ptA, None, pos_idsA)

    print("Compare (PT_origin) vs (PT_test):")
    compare_two_model_outputs(pt_attn_originA, pt_attn_testA, x_ptA, position_ids=pos_idsA,
                              labelA="PT_origin", labelB="PT_test")
    print("Compare (PT_origin) vs (FL):")
    compare_pt_with_fl(pt_attn_originA, fl_attnA, fl_varsA["params"], x_ptA, position_ids=pos_idsA,
                       labelPT="PT_origin", labelFL="FL")
    print("Compare (PT_test)   vs (FL) => if we want to see if PT_test can also match:")
    rngA2 = jax.random.PRNGKey(9)
    fl_varsA2 = load_pytorch_weights_into_flax_attention(pt_attn_testA, fl_attnA, rngA2, x_ptA, None, pos_idsA)
    compare_pt_with_fl(pt_attn_testA, fl_attnA, fl_varsA2["params"], x_ptA, position_ids=pos_idsA,
                       labelPT="PT_test", labelFL="FL")


    # ----------------------------------------------------
    # scenario B) flexible=True, rope_type="default"
    # ----------------------------------------------------
    print("\n--- Scenario B: flexible=True, rope_type=default ---")

    configB = DummyMoshiConfig(hidden_size=16, num_attention_heads=4, head_dim=4,
                               rope_scaling=None, rope_theta=10000., num_codebooks=2)
    pt_attn_originB = MoshiAttentionPT_origin(configB, use_rope=True, use_flexible_linear=True).eval()
    pt_attn_testB = MoshiAttentionPT(
        hidden_size=16,
        num_heads=4,
        head_dim=4,
        num_codebooks=2,
        use_flexible_linear=True,
        rope=True,
        rope_theta=10000.0,
        rope_type="default"
    ).eval()
    fl_attnB = MoshiAttentionFL(
        hidden_size=16, num_heads=4, head_dim=4,
        num_codebooks=2, use_flexible_linear=True,
        rope=True, rope_theta=10000., rope_type="default"
    )

    B2,S2=3,2
    x_ptB = torch.randn(B2,S2,16)
    pos_idsB = torch.tensor([[0,1],[2,5],[3,7]], dtype=torch.long)

    rngB = jax.random.PRNGKey(100)
    fl_varsB = load_pytorch_weights_into_flax_attention(pt_attn_originB, fl_attnB, rngB, x_ptB, None, pos_idsB)

    print("Compare (PT_origin) vs (PT_test):")
    compare_two_model_outputs(pt_attn_originB, pt_attn_testB, x_ptB, position_ids=pos_idsB,
                              labelA="PT_origin", labelB="PT_test")
    print("Compare (PT_origin) vs (FL):")
    compare_pt_with_fl(pt_attn_originB, fl_attnB, fl_varsB["params"], x_ptB, position_ids=pos_idsB,
                       labelPT="PT_origin", labelFL="FL")
    print("Compare (PT_test) vs (FL):")
    rngB2 = jax.random.PRNGKey(200)
    fl_varsB2 = load_pytorch_weights_into_flax_attention(pt_attn_testB, fl_attnB, rngB2, x_ptB, None, pos_idsB)
    compare_pt_with_fl(pt_attn_testB, fl_attnB, fl_varsB2["params"], x_ptB, position_ids=pos_idsB,
                       labelPT="PT_test", labelFL="FL")

    print("\n==== Done. Check the diffs above (ideally ~1e-6 or smaller) ====")


if __name__=="__main__":
    main()
