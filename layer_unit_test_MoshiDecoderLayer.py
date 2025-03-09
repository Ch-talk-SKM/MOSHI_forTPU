#!/usr/bin/env python3
# coding=utf-8

"""
layer_unit_test_MoshiDecoderLayer.py

- We unify PyTorch (PT) and Flax (FL) MoshiDecoderLayer by importing
  the official MoshiAttentionPT from layer_unit_test_TPU_att_rope_compare3.py,
  ensuring the same SCALED DOT-PRODUCT ATTENTION + RoPE logic in both PT and FL.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import flax
from flax import linen as fnn

# ============= External imports =============
from layer_unit_test_TPU_compare3 import (
    MoshiRMSNormPT,       # PyTorch RMSNorm
    MoshiRMSNormFL,       # Flax RMSNorm
    load_pt_weights_gatingmlp,
    MoshiGatingMLPFL,     # Flax version of gating MLP
)
from layer_unit_test_TPU_att_rope_compare3 import (
    # We import the "official" MoshiAttentionPT (scaled-dot-product attention logic)
    MoshiAttentionPT,
    MoshiAttentionFL,
    MoshiFlexibleLinearPT, 
    MoshiLinearPT,
    MoshiLinearFL,
    MoshiFlexibleLinearFL,
    load_pytorch_weights_into_flax_attention,
)

# -------------------------------------------------------------------------
# 1) PyTorch GatingMLP: local class (optionally we could also import if exist)
# -------------------------------------------------------------------------
class MoshiGatingMLPPT(nn.Module):
    """
    PT gating MLP with layer_idx support.
    """
    def __init__(self, hidden_size, ffn_dim, num_codebooks=1, hidden_act="relu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_codebooks = num_codebooks
        self.hidden_act = hidden_act

        if num_codebooks <= 1:
            self.fc1 = nn.Linear(hidden_size, ffn_dim, bias=False)
            self.fc2 = nn.Linear(ffn_dim // 2, hidden_size, bias=False)
        else:
            self.fc1 = MoshiFlexibleLinearPT(hidden_size, ffn_dim, num_codebooks)
            self.fc2 = MoshiFlexibleLinearPT(ffn_dim // 2, hidden_size, num_codebooks)

        if hidden_act == "relu":
            self.activation_fn = nn.ReLU()
        elif hidden_act == "gelu":
            self.activation_fn = nn.GELU()
        else:
            self.activation_fn = nn.ReLU()

    def forward(self, x: torch.Tensor, layer_idx: Optional[Union[int, torch.Tensor]] = None):
        # fc1
        if self.num_codebooks <= 1:
            x = self.fc1(x)
        else:
            # optionally pass layer_idx if needed by MoshiFlexibleLinearPT
            x = self.fc1(x, layer_idx=layer_idx)

        B, S, D = x.shape
        x = x.view(B, S, 2, -1)
        gate = self.activation_fn(x[..., 0, :])
        val  = x[..., 1, :]
        x = gate * val

        if self.num_codebooks <= 1:
            x = self.fc2(x)
        else:
            x = self.fc2(x, layer_idx=layer_idx)
        return x


# -------------------------------------------------------------------------
# 2) PyTorch MoshiDecoderLayerPT
#    - uses official MoshiAttentionPT from layer_unit_test_TPU_att_rope_compare3.py
# -------------------------------------------------------------------------
class MoshiDecoderLayerPT(nn.Module):
    """
    PyTorch decoder layer that uses:
      - self_attn = MoshiAttentionPT (now with layer_idx)
      - MLP = MoshiGatingMLPPT (can also accept layer_idx)
      - RMSNorm = MoshiRMSNormPT
    """
    def __init__(self, config, layer_idx: int, use_flexible_linear: bool, use_rope: bool = True):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.use_flexible_linear = use_flexible_linear

        # Self-Attn (MoshiAttentionPT) - now supports layer_idx
        self.self_attn = MoshiAttentionPT(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_codebooks=config.num_codebooks,
            use_flexible_linear=use_flexible_linear,
            rope=use_rope,
            rope_theta=getattr(config, "rope_theta", 10000.0),
            rope_type="default",  # 혹은 config.rope_scaling["rope_type"] 등
        )

        # MLP (Gating)
        self.mlp = MoshiGatingMLPPT(
            hidden_size=config.hidden_size,
            ffn_dim=config.ffn_dim,
            num_codebooks=(config.num_codebooks if use_flexible_linear else 1),
            hidden_act=config.hidden_act
        )

        # RMSNorm
        self.input_layernorm = MoshiRMSNormPT(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MoshiRMSNormPT(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_idx_attn: Optional[Union[int, torch.Tensor]] = None,
        layer_idx_mlp: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        We pass layer_idx_attn into self_attn, and layer_idx_mlp into MLP,
        enabling flexible linear usage for both.
        """
        # 1) Pre-norm + self-attn
        residual = hidden_states
        x = self.input_layernorm(hidden_states)

        # Self-attn with potential flexible usage
        x = self.self_attn(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_idx=layer_idx_attn,
        )
        x = residual + x

        # 2) Post-attn norm + MLP
        residual = x
        x2 = self.post_attention_layernorm(x)

        # Gating MLP with potential flexible usage
        x2 = self.mlp(x2, layer_idx=layer_idx_mlp)
        out = residual + x2

        return out

# -------------------------------------------------------------------------
# 3) Flax MoshiDecoderLayerFL
# -------------------------------------------------------------------------
class MoshiDecoderLayerFL(fnn.Module):
    config: Any
    use_flexible_linear: bool = False
    use_rope: bool = True

    def setup(self):
        self.input_layernorm = MoshiRMSNormFL(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            name="input_layernorm"
        )
        self.post_attention_layernorm = MoshiRMSNormFL(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            name="post_attention_layernorm"
        )

        rope_type = (
            self.config.rope_scaling["rope_type"]
            if getattr(self.config, "rope_scaling", None)
            else "default"
        )
        # imported MoshiAttentionFL (which does accept layer_idx)
        self.self_attn = MoshiAttentionFL(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            head_dim=self.config.head_dim,
            num_codebooks=(self.config.num_codebooks if self.use_flexible_linear else 1),
            use_flexible_linear=self.use_flexible_linear,
            rope=self.use_rope,
            rope_theta=getattr(self.config, "rope_theta", 10000.0),
            rope_type=rope_type,
            name="self_attn",
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
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        layer_idx_attn: Optional[Union[int,jnp.ndarray]]=None,
        layer_idx_mlp: Optional[Union[int,jnp.ndarray]]=None
    ) -> jnp.ndarray:
        x = self.input_layernorm(hidden_states)
        # MoshiAttentionFL does have layer_idx => pass layer_idx_attn
        attn_out = self.self_attn(
            x, attention_mask=attention_mask,
            position_ids=position_ids, layer_idx=layer_idx_attn
        )
        x = hidden_states + attn_out

        x2 = self.post_attention_layernorm(x)
        mlp_out = self.mlp(x2, layer_idx=layer_idx_mlp)
        out = x + mlp_out
        return out


# -------------------------------------------------------------------------
# 4) PT->FL param loading + compare
# -------------------------------------------------------------------------
def merge_params_dict(base_params, update_dict):
    newp = copy.deepcopy(base_params)
    for k,v in update_dict.items():
        if isinstance(v, dict) and k in newp:
            newp[k] = merge_params_dict(newp[k], v)
        else:
            newp[k] = v
    return newp


def load_decoder_layer_pt_to_flax(
    pt_layer,
    fl_layer,
    rng,
    sample_hidden_states,
    attention_mask=None,
    position_ids=None,
):
    """
    PyTorch MoshiDecoderLayer -> Flax MoshiDecoderLayer param copy
    - self_attn, mlp => use dummy submodules + load_pytorch_weights_into_flax_attention / load_pt_weights_gatingmlp
    """
    from layer_unit_test_TPU_att_rope_compare3 import MoshiAttentionFL, load_pytorch_weights_into_flax_attention
    from layer_unit_test_TPU_compare3 import MoshiGatingMLPFL, load_pt_weights_gatingmlp

    h_np = sample_hidden_states.detach().cpu().numpy()
    am_np = attention_mask.detach().cpu().numpy() if attention_mask is not None else None
    pi_np = position_ids.detach().cpu().numpy() if position_ids is not None else None

    init_vars = fl_layer.init(
        rng,
        jnp.array(h_np),
        None if am_np is None else jnp.array(am_np),
        None if pi_np is None else jnp.array(pi_np),
    )
    fl_params = copy.deepcopy(init_vars["params"])

    # RMSNorm
    pt_rms_in = pt_layer.input_layernorm
    fl_params["input_layernorm"]["weight"] = jnp.array(pt_rms_in.weight.detach().cpu().numpy())

    pt_rms_post = pt_layer.post_attention_layernorm
    fl_params["post_attention_layernorm"]["weight"] = jnp.array(pt_rms_post.weight.detach().cpu().numpy())

    # Self-Attention (dummy submodule usage)
    dummy_fl_attn = MoshiAttentionFL(
        hidden_size=pt_layer.config.hidden_size,
        num_heads=pt_layer.config.num_attention_heads,
        head_dim=pt_layer.config.head_dim,
        num_codebooks=(pt_layer.config.num_codebooks if pt_layer.use_flexible_linear else 1),
        use_flexible_linear=pt_layer.use_flexible_linear,
        rope=getattr(pt_layer, "use_rope", True),
        rope_theta=getattr(pt_layer.config, "rope_theta", 10000.0),
        rope_type="default",
        name="self_attn_dummy"
    )
    dummy_init_attn = dummy_fl_attn.init(
        rng,
        jnp.array(h_np),
        None if am_np is None else jnp.array(am_np),
        None if pi_np is None else jnp.array(pi_np),
    )
    fl_attn_params_dict = load_pytorch_weights_into_flax_attention(
        pt_layer.self_attn,  # note: self.self_attn doesn't accept layer_idx, but param is the same
        dummy_fl_attn,
        rng,
        sample_hidden_states,
        attention_mask,
        position_ids
    )
    fl_params["self_attn"] = merge_params_dict(
        fl_params["self_attn"],
        fl_attn_params_dict["params"]
    )

    # MLP
    dummy_fl_mlp = MoshiGatingMLPFL(
        hidden_size=pt_layer.config.hidden_size,
        ffn_dim=pt_layer.config.ffn_dim,
        num_codebooks=(pt_layer.config.num_codebooks if pt_layer.use_flexible_linear else 1),
        hidden_act=pt_layer.config.hidden_act,
        name="mlp_dummy"
    )
    dummy_init_mlp = dummy_fl_mlp.init(
        rng,
        jnp.array(h_np),
    )
    fl_mlp_params_dict = load_pt_weights_gatingmlp(
        pt_layer.mlp,
        dummy_fl_mlp,
        rng,
        sample_hidden_states
    )
    fl_params["mlp"] = merge_params_dict(
        fl_params["mlp"],
        fl_mlp_params_dict["params"]
    )

    final_params = copy.deepcopy(init_vars)
    final_params["params"] = merge_params_dict(fl_params, {})
    return final_params


def compare_decoder_layers_pt_fl(
    pt_layer,
    fl_layer,
    fl_params,
    hidden_states_pt,
    attention_mask=None,
    position_ids=None,
    label_pt="PT",
    label_fl="FL",
    layer_idx_attn=None,
    layer_idx_mlp=None
):
    import numpy as np
    with torch.no_grad():
        # "MoshiAttentionPT.forward()" doesn't accept layer_idx => so let's ignore layer_idx_attn
        out_pt = pt_layer(
            hidden_states_pt,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_idx_attn=layer_idx_attn,
            layer_idx_mlp=layer_idx_mlp
        )
    out_pt_np = out_pt.detach().cpu().numpy()

    # Convert layer_idx_mlp from torch->jax if needed
    if isinstance(layer_idx_mlp, torch.Tensor):
        layer_idx_mlp = jax.numpy.array(layer_idx_mlp.detach().cpu().numpy(), dtype=jax.numpy.int32)

    am_jnp = None if attention_mask is None else jnp.array(attention_mask.detach().cpu().numpy())
    pi_jnp = None if position_ids is None else jnp.array(position_ids.detach().cpu().numpy())

    out_fl = fl_layer.apply(
        {"params": fl_params["params"]},
        jnp.array(hidden_states_pt.detach().cpu().numpy()),
        am_jnp,
        pi_jnp,
        layer_idx_attn=layer_idx_attn,  # FL does accept layer_idx_attn
        layer_idx_mlp=layer_idx_mlp
    )
    out_fl_np = np.array(out_fl)

    diff = np.abs(out_pt_np - out_fl_np).mean()
    print(f"  {label_pt} vs {label_fl}, layer_idx_attn={layer_idx_attn}, layer_idx_mlp={layer_idx_mlp}, diff={diff:.6e}")
    return diff


def main_decoder_layer_test():
    print("=== Compare MoshiDecoderLayer PT vs FL with layer_idx variations ===")

    class DummyConfig:
        hidden_size = 16
        num_attention_heads = 4
        head_dim = 4
        num_codebooks = 5
        ffn_dim = 32
        hidden_act = "gelu"
        rope_theta = 10000.0
        rope_scaling = None
        rms_norm_eps = 1e-6

    config = DummyConfig()
    layer_idx = 0
    use_flexible_linear = True
    use_rope = True

    # Create PT / FL layer
    # PT layer: *cannot* handle layer_idx_attn in attention (MoshiAttentionPT has no 'layer_idx' param)
    pt_layer = MoshiDecoderLayerPT(config, layer_idx, use_flexible_linear, use_rope).eval()
    # FL layer: can handle layer_idx_attn
    fl_layer = MoshiDecoderLayerFL(config, use_flexible_linear, use_rope)

    # random input
    B, S = 2, 5
    hidden_states_pt = torch.randn(B, S, config.hidden_size)

    # WARNING: random mask => big difference in softmax
    attention_mask_pt = torch.randn(B, 1, S, S)
    #attention_mask_pt = torch.zeros(B, 1, S, S)  # no mask


    position_ids_pt = torch.tensor([[0,1,2,3,4],[10,11,12,13,14]], dtype=torch.long)

    # init
    rng = jax.random.PRNGKey(0)
    fl_init_params = fl_layer.init(
        rng,
        jnp.array(hidden_states_pt.detach().cpu().numpy()),
        jnp.array(attention_mask_pt.detach().cpu().numpy()),
        jnp.array(position_ids_pt.detach().cpu().numpy())
    )

    # PT->FL param load
    rng2 = jax.random.PRNGKey(123)
    fl_loaded = load_decoder_layer_pt_to_flax(
        pt_layer, fl_layer, rng2,
        hidden_states_pt, attention_mask_pt, position_ids_pt
    )

    # Scenarios
    scenario_list = [
        (None, None),
        (0, 1),
        (torch.tensor([0,1,0,1,0], dtype=torch.long),
         torch.tensor([1,1,0,0,1], dtype=torch.long)),
    ]

    for (lia, lim) in scenario_list:
        # uninit
        compare_decoder_layers_pt_fl(
            pt_layer, fl_layer, fl_init_params,
            hidden_states_pt, attention_mask_pt, position_ids_pt,
            label_pt="PT(uninit)", label_fl="FL(uninit)",
            layer_idx_attn=lia, layer_idx_mlp=lim
        )
        # loaded
        compare_decoder_layers_pt_fl(
            pt_layer, fl_layer, fl_loaded,
            hidden_states_pt, attention_mask_pt, position_ids_pt,
            label_pt="PT(loaded)", label_fl="FL(loaded)",
            layer_idx_attn=lia, layer_idx_mlp=lim
        )

    print("=== Done ===")


if __name__ == "__main__":
    main_decoder_layer_test()
