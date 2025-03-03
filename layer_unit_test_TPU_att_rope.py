#!/usr/bin/env python3
# coding=utf-8

"""
layer_unit_test_TPU_att_rope.py

- PyTorch + Flax code for MoshiAttention with RoPE,
  ensuring consistent shape usage:
  (1, S, head_dim) for cos,sin, then rotate_half(q).
  This avoids the size mismatch (4 vs 2) and keeps the RoPE math correct.

- PyTorch -> Flax weight transfer and output comparison to verify correctness.
"""

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
# Dummy config & rope init
###############################################################################
def rope_init_default(config, device=None, seq_len=None):
    """
    (basic version) inv_freq = 1/(10000^(range(0,dim,2)/dim)), attention_scaling=1.0
    """
    hidden_dim = config.hidden_size // config.num_attention_heads
    inv_freq = torch.tensor(
        1.0 / (10000.0 ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim)),
        device=device
    )
    attention_scaling = 1.0
    return inv_freq, attention_scaling

def rope_init_dynamic(config, device=None, seq_len=None):
    """
    (dynamic version) inv_freq = 1/(rope_theta^(range(0,dim,2)/dim)), and
    attention_scaling depends on seq_len^( -0.05 ).
    """
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
        rope_theta=10000.0
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta


###############################################################################
# 1) PyTorch modules
###############################################################################
def rotate_half(x):
    """
    x shape: (..., head_dim=4).
    half = 2 -> x1= x[..., :2], x2= x[...,2:]
    return cat([-x2, x1], dim=-1), still shape (...,4).
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    q,k: shape (B, nHeads, S, head_dim)
    cos,sin: shape (1, S, head_dim) -> unsqueeze -> (1, 1, S, head_dim)
    => final broadcast is (B,nHeads,S,head_dim) vs (1,1,S,head_dim).
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # (1,1,S,head_dim)
    sin = sin.unsqueeze(unsqueeze_dim)  # (1,1,S,head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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
        """
        Generate cos,sin with final shape (1,S, head_dim),
        so that (q * cos) has no mismatch with q: (B,nHeads,S,head_dim).
        """
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        bsz, s_len = position_ids.shape
        p_float = position_ids.float()
        inv_f = self.inv_freq.float()
        hidden_dim = self.config.hidden_size // self.config.num_attention_heads

        # freqs => shape (S, head_dim//2)
        freqs = torch.einsum("i,d->id", p_float[0], inv_f)
        cos_half = freqs.cos() * self.attention_scaling
        sin_half = freqs.sin() * self.attention_scaling

        # 2배 확장 -> (S,4)
        cos = torch.cat([cos_half, cos_half], dim=-1)
        sin = torch.cat([sin_half, sin_half], dim=-1)

        # => (1,S,4)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        return cos, sin

def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
    """
    For certain cases we might replicate k,v across multiple heads,
    but here n_rep=1 => no effect.
    """
    b, n, s, h = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hs = hidden_states[:, :, None, :, :].expand(b, n, n_rep, s, h)
    return hs.reshape(b, n*n_rep, s, h)

class MoshiLinearPT(pt_nn.Module):
    def __init__(self, input_dim, output_dim, num_codebooks, use_flexible_linear=False):
        super().__init__()
        self.use_flexible_linear = use_flexible_linear
        if not use_flexible_linear:
            self.linear = pt_nn.Linear(input_dim, output_dim, bias=False)
        else:
            pass  # flexible not essential for example

    def forward(self, x, layer_idx=None):
        return self.linear(x)

class MoshiAttentionPT(pt_nn.Module):
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
            rope_scaling={"rope_type": rope_type} if rope_type!="default" else None,
            rope_theta=rope_theta
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

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz,q_len,_ = hidden_states.size()
        q = self.q_proj(hidden_states)  # (B,S,num_heads*head_dim)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # => (B, num_heads, S, head_dim)
        q = q.view(bsz,q_len,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(bsz,q_len,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(bsz,q_len,self.num_heads,self.head_dim).transpose(1,2)

        if self.use_rope and self.rotary_emb is not None and position_ids is not None:
            cos,sin = self.rotary_emb(v, position_ids)  # shape (1,S,4)
            q,k = apply_rotary_pos_emb(q,k,cos,sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_weights = torch.matmul(q, k.transpose(2,3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v).transpose(1,2).contiguous()
        attn_output = attn_output.view(bsz,q_len,-1)
        attn_output = self.o_proj(attn_output)
        return attn_output


###############################################################################
# 2) Flax modules
###############################################################################
def rotate_half_fl(x):
    """
    x shape: (..., head_dim=4).
    split -> x1= x[..., :2], x2= x[..., 2:]
    => return cat([-x2, x1], axis=-1), shape(...,4).
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb_fl(q, k, cos, sin):
    """
    cos,sin => shape (1,S,4),
    => expand_dims => (1,1,S,4).
    q,k => (B,nHeads,S,4).
    Then q_embed= q*cos + rotate_half_fl(q)* sin, etc.
    """
    cos = jnp.expand_dims(cos, axis=1)  # => (1,1,S,4)
    sin = jnp.expand_dims(sin, axis=1)

    q_embed = (q * cos) + (rotate_half_fl(q) * sin)
    k_embed = (k * cos) + (rotate_half_fl(k) * sin)
    return q_embed, k_embed

def rope_init_fl(config: DummyMoshiConfig, seq_len=None):
    """
    Flax version of rope init. Return np.array inv_freq, float attention_scaling.
    """
    rope_type = "default"
    if getattr(config,"rope_scaling",None):
        rope_type = config.rope_scaling.get("rope_type","default")
    hidden_dim = config.hidden_size // config.num_attention_heads

    if rope_type=="dynamic":
        if seq_len is None:
            seq_len = config.max_position_embeddings
        inv_freq = np.array(
            1.0/(config.rope_theta**(
                np.arange(0, hidden_dim,2)/hidden_dim
            )), dtype=np.float32
        )
        attention_scaling = 1.0/(float(seq_len)**0.05)
    else:
        inv_freq = np.array(
            1.0/(config.rope_theta**(
                np.arange(0, hidden_dim,2)/hidden_dim
            )), dtype=np.float32
        )
        attention_scaling=1.0
    return inv_freq, attention_scaling

class MoshiRotaryEmbeddingFL(nn.Module):
    config: DummyMoshiConfig

    @nn.compact
    def __call__(self, x, position_ids):
        # 1) compute freqs, shape(S, head_dim//2)
        seq_len = jnp.max(position_ids) + 1
        inv_freq, scaling = rope_init_fl(self.config, seq_len=seq_len)
        p32 = position_ids.astype(jnp.float32)
        freqs = jnp.einsum("i,d->id", p32[0], jnp.array(inv_freq,dtype=jnp.float32))

        cos_half = jnp.cos(freqs)*scaling
        sin_half = jnp.sin(freqs)*scaling

        # 2) double expansion -> (S, head_dim=4)
        cos = jnp.concatenate([cos_half, cos_half], axis=-1)
        sin = jnp.concatenate([sin_half, sin_half], axis=-1)

        # => (1,S,4)
        cos = cos[None, ...]
        sin = sin[None, ...]
        return cos, sin

class MoshiLinearFL(nn.Module):
    input_dim: int
    output_dim: int
    num_codebooks: int
    use_flexible_linear: bool=False

    @nn.compact
    def __call__(self, x, layer_idx=None):
        # Simple version: just a single kernel
        kernel = self.param(
            "kernel",
            nn.initializers.normal(stddev=1.0),
            (self.input_dim, self.output_dim)
        )
        if x.ndim==2:
            out = jnp.dot(x, kernel)      # shape (B, output_dim)
        else:
            out = jnp.einsum("bsd,do->bso", x, kernel)
        return out

def repeat_kv_fl(hidden_states: jnp.ndarray, n_rep: int):
    """
    For repeating K,V if needed. Here n_rep=1 => no effect.
    """
    B,N,S,H = hidden_states.shape
    if n_rep==1:
        return hidden_states
    tile = jnp.tile(hidden_states[:, :, None, :, :], (1,1,n_rep,1,1))
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
    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        B,S,D = hidden_states.shape

        q_proj = MoshiLinearFL(self.hidden_size, self.num_heads*self.head_dim, self.num_codebooks, self.use_flexible_linear, name="q_proj")
        k_proj = MoshiLinearFL(self.hidden_size, self.num_heads*self.head_dim, self.num_codebooks, self.use_flexible_linear, name="k_proj")
        v_proj = MoshiLinearFL(self.hidden_size, self.num_heads*self.head_dim, self.num_codebooks, self.use_flexible_linear, name="v_proj")
        o_proj = MoshiLinearFL(self.num_heads*self.head_dim, self.hidden_size, self.num_codebooks, self.use_flexible_linear, name="o_proj")

        # shape => (B,S, num_heads*head_dim) -> reshape -> (B, nHeads, S, head_dim)
        q = q_proj(hidden_states).reshape((B,S,self.num_heads,self.head_dim)).transpose((0,2,1,3))
        k = k_proj(hidden_states).reshape((B,S,self.num_heads,self.head_dim)).transpose((0,2,1,3))
        v = v_proj(hidden_states).reshape((B,S,self.num_heads,self.head_dim)).transpose((0,2,1,3))

        if self.rope and position_ids is not None:
            config = DummyMoshiConfig(
                hidden_size = self.hidden_size,
                num_attention_heads = self.num_heads,
                rope_scaling = {"rope_type": self.rope_type} if self.rope_type!="default" else None,
                rope_theta = self.rope_theta,
            )
            cos,sin = MoshiRotaryEmbeddingFL(config)(v, position_ids)  # (1,S,4)
            q,k = apply_rotary_pos_emb_fl(q, k, cos, sin)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k)*scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.softmax(attn_weights, axis=-1)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        out = out.transpose((0,2,1,3)).reshape((B,S, self.num_heads*self.head_dim))
        out = o_proj(out)
        return out


###############################################################################
# 3) load PyTorch -> Flax, compare
###############################################################################
def load_pytorch_weights_into_flax_attention(pt_attn, flax_attn, rng, sample_input, attention_mask=None, position_ids=None):
    """
    1) init flax_attn to get param dict,
    2) copy PyTorch param => Flax param,
    3) return new param dict
    """
    sample_input_jnp = jnp.array(sample_input.detach().cpu().numpy())
    attn_jnp = None
    pos_jnp = None
    if attention_mask is not None:
        attn_jnp = jnp.array(attention_mask.detach().cpu().numpy())
    if position_ids is not None:
        pos_jnp = jnp.array(position_ids.detach().cpu().numpy())

    variables = flax_attn.init(rng, sample_input_jnp, attn_jnp, pos_jnp)
    new_params = dict(variables["params"])

    for proj_name in ["q_proj","k_proj","v_proj","o_proj"]:
        pt_layer = getattr(pt_attn, proj_name)
        fl_layer = new_params[proj_name]
        pt_w = pt_layer.linear.weight.detach().cpu().numpy()  # shape (out_dim, in_dim)
        # In Flax: kernel is shape (in_dim, out_dim) => need transpose
        fl_layer["kernel"] = jnp.array(pt_w.T)
        new_params[proj_name] = fl_layer

    return {"params": new_params}

def compare_attention_outputs(pt_attn, jax_attn, jax_params, hidden_states_pt, attention_mask_pt=None, position_ids_pt=None):
    with torch.no_grad():
        out_pt = pt_attn(hidden_states_pt, attention_mask=attention_mask_pt, position_ids=position_ids_pt)
    out_pt_np = out_pt.detach().cpu().numpy()

    hs_fl = jnp.array(hidden_states_pt.detach().cpu().numpy())
    am_fl=None
    pi_fl=None
    if attention_mask_pt is not None:
        am_fl = jnp.array(attention_mask_pt.detach().cpu().numpy())
    if position_ids_pt is not None:
        pi_fl = jnp.array(position_ids_pt.detach().cpu().numpy())

    out_fl = jax_attn.apply({"params": jax_params}, hs_fl, am_fl, pi_fl)
    out_fl_np = np.array(out_fl)

    diff = np.abs(out_pt_np - out_fl_np).mean()
    print(f"[compare_attention_outputs] mean diff = {diff:.6f}")
    return diff


###############################################################################
# 4) Demo main
###############################################################################
def main():
    hidden_size = 16
    num_heads = 4
    head_dim = 4
    rope_type = "dynamic"

    # 1) PyTorch model
    pt_attn = MoshiAttentionPT(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_codebooks=1,
        use_flexible_linear=False,
        rope=True,
        rope_theta=10000.0,
        rope_type=rope_type
    ).eval()

    # 2) sample input
    B,S = 2,7
    x_pt = torch.randn(B,S,hidden_size)
    attn_mask_pt = None
    pos_ids_pt = torch.tensor([[0,1,2,3,10,11,14],[0,1,2,9,10,13,14]],dtype=torch.long)

    # 3) Flax model
    flax_attn = MoshiAttentionFL(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_codebooks=1,
        use_flexible_linear=False,
        rope=True,
        rope_theta=10000.0,
        rope_type=rope_type
    )

    # 4) load PyTorch -> Flax, compare
    rng = jax.random.PRNGKey(0)
    flax_vars = load_pytorch_weights_into_flax_attention(
        pt_attn, flax_attn, rng, x_pt, attn_mask_pt, pos_ids_pt
    )
    diff = compare_attention_outputs(pt_attn, flax_attn, flax_vars["params"], x_pt, attn_mask_pt, pos_ids_pt)
    print(f"Final diff = {diff:.6f}")


if __name__=="__main__":
    main()
