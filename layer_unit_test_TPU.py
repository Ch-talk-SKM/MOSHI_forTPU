#!/usr/bin/env python3
# coding=utf-8

"""
Final unified code that:
1) Defines PyTorch & Flax classes for MoshiFlexibleLinear, MoshiRMSNorm, MoshiGatingMLP.
2) Ensures all 'forward' / '__call__' can optionally take `layer_idx=None`.
3) Compares outputs from PT -> Flax, ignoring or using layer_idx consistently.

No more TypeError about extra arguments in RMSNorm.
No more attribute errors about .unfreeze().
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

################################################################################
# 1) PyTorch modules
################################################################################

class MoshiFlexibleLinearPT(pt_nn.Module):
    """
    Flexible linear layer (PyTorch) with optional layer_idx.
    """
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        self.num_layers = num_layers
        # (num_layers, out_dim, in_dim)
        self.weight = pt_nn.Parameter(torch.randn(num_layers, output_size, input_size))

    def forward(self, x, layer_idx=None):
        """
        x: shape (B, S, in_dim) or (B, in_dim)
        layer_idx: None or int or 1D
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # => (B,1,D)
        B, S, D = x.shape
        nl, out_dim, in_dim = self.weight.shape

        if D != in_dim:
            raise ValueError(f"[MoshiFlexibleLinearPT] input dim mismatch: x has {D}, w expects {in_dim}.")

        # case 1) layer_idx=None => must have S==num_layers
        if layer_idx is None:
            if S != nl:
                raise ValueError("S!=num_layers in PT flexible linear usage.")
            outs = []
            for i in range(S):
                w_i = self.weight[i]
                outs.append(F.linear(x[:, i], w_i))
            return torch.stack(outs, dim=1)

        # case 2) layer_idx=int => single weight for all S
        if isinstance(layer_idx, int):
            if not (0 <= layer_idx < nl):
                raise ValueError("invalid layer_idx in PT flexible linear usage.")
            w = self.weight[layer_idx]
            return F.linear(x, w)  # shape (B,S,out_dim)

        # case 3) layer_idx is 1D
        if layer_idx.dim() == 1:
            L = layer_idx.shape[0]
            if S == L:
                outs = []
                for i in range(L):
                    outs.append(F.linear(x[:, i], self.weight[layer_idx[i]]))
                return torch.stack(outs, dim=1)
            if S == 1:
                x_squeezed = x[:, 0]
                outs = []
                for i in layer_idx:
                    outs.append(F.linear(x_squeezed, self.weight[i]))
                return torch.stack(outs, dim=1)
            raise ValueError("shape mismatch in PT flexible linear usage (layer_idx).")

        raise ValueError("layer_idx must be None/int/1D tensor in PT flexible linear usage.")


class MoshiRMSNormPT(pt_nn.Module):
    """
    PyTorch RMSNorm with optional `layer_idx=None`.
    We ignore layer_idx for RMSNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = pt_nn.Parameter(torch.ones(dim))

    def forward(self, x, layer_idx=None):
        # just ignore layer_idx
        x_f32 = x.float()
        rms = torch.sqrt((x_f32**2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x_f32 / rms
        scaled = normed * self.weight.float()
        return scaled.type_as(x)


_ACT2FN = {
    "relu": pt_nn.ReLU(),
    "gelu": pt_nn.GELU(),
}


class MoshiGatingMLPPT(pt_nn.Module):
    """
    Gating MLP in PyTorch with optional layer_idx.
    """
    def __init__(self, hidden_size, ffn_dim, num_codebooks=1, hidden_act="relu"):
        super().__init__()
        if ffn_dim < 2 or ffn_dim % 2 != 0:
            raise ValueError("ffn_dim must be even and >=2.")
        self.hidden_act = hidden_act
        self.activation_fn = _ACT2FN.get(hidden_act, pt_nn.ReLU())
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_codebooks = num_codebooks

        if num_codebooks <= 1:
            self.fc1 = pt_nn.Linear(hidden_size, ffn_dim, bias=False)
            self.fc2 = pt_nn.Linear(ffn_dim // 2, hidden_size, bias=False)
        else:
            self.fc1 = MoshiFlexibleLinearPT(hidden_size, ffn_dim, num_codebooks)
            self.fc2 = MoshiFlexibleLinearPT(ffn_dim // 2, hidden_size, num_codebooks)

    def forward(self, x, layer_idx=None):
        # 1) fc1
        if self.num_codebooks <= 1:
            x = self.fc1(x)
        else:
            x = self.fc1(x, layer_idx=layer_idx)

        B, S, F = x.shape
        x = x.view(B, S, 2, -1)
        gate = self.activation_fn(x[..., 0, :])
        val  = x[..., 1, :]
        x = gate * val
        # 2) fc2
        if self.num_codebooks <= 1:
            x = self.fc2(x)
        else:
            x = self.fc2(x, layer_idx=layer_idx)
        return x


################################################################################
# 2) Flax modules
################################################################################

class MoshiFlexibleLinearFL(nn.Module):
    """
    Flax flexible linear with optional layer_idx.
    """
    input_size: int
    output_size: int
    num_layers: int

    @nn.compact
    def __call__(self, x, layer_idx=None):
        # param shape (num_layers, out_dim, in_dim)
        weight = self.param("weight",
                            nn.initializers.normal(stddev=1.0),
                            (self.num_layers, self.output_size, self.input_size))
        if x.ndim == 2:
            x = x[:, None, :]
        B, S, D = x.shape
        if D != self.input_size:
            raise ValueError("dimension mismatch in Flax flexible linear")

        def matmul_2d(v, w):
            # v: (B, in_dim)
            # w: (out_dim, in_dim)
            return jnp.einsum("bd,od->bo", v, w)

        # case 1) layer_idx=None => S==num_layers
        if layer_idx is None:
            if S != self.num_layers:
                raise ValueError("Flax flexible: S != num_layers")
            outs = []
            for i in range(S):
                w_i = weight[i]
                outs.append(matmul_2d(x[:, i], w_i))
            return jnp.stack(outs, axis=1)

        # case 2) layer_idx=int => single weight
        if isinstance(layer_idx, int):
            if not (0 <= layer_idx < self.num_layers):
                raise ValueError("invalid layer_idx in flax flexible")
            w = weight[layer_idx]
            outs = []
            for i in range(S):
                outs.append(matmul_2d(x[:, i], w))
            return jnp.stack(outs, axis=1)

        # case 3) layer_idx=1D jnp array
        if layer_idx is not None and layer_idx.ndim == 1:
            L = layer_idx.shape[0]
            if S == L:
                outs = []
                for i in range(L):
                    w_i = weight[layer_idx[i]]
                    outs.append(matmul_2d(x[:, i], w_i))
                return jnp.stack(outs, axis=1)
            if S == 1:
                x_squeezed = x[:, 0]
                outs = []
                for i in layer_idx:
                    outs.append(matmul_2d(x_squeezed, weight[i]))
                return jnp.stack(outs, axis=1)
            raise ValueError("flax flexible shape mismatch for layer_idx")
        raise ValueError("bad layer_idx in flax flexible")


class MoshiRMSNormFL(nn.Module):
    """
    Flax RMSNorm with optional layer_idx (ignored).
    """
    dim: int
    eps: float=1e-6

    @nn.compact
    def __call__(self, x, layer_idx=None):
        # ignore layer_idx
        weight = self.param("weight", nn.initializers.ones, (self.dim,))
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        normed = x_f32 / rms
        scaled = normed * weight
        return scaled.astype(x.dtype)


class MoshiGatingMLPFL(nn.Module):
    """
    Flax gating MLP with optional layer_idx usage if num_codebooks>1.
    """
    hidden_size: int
    ffn_dim: int
    num_codebooks: int=1
    hidden_act: str="relu"

    @nn.compact
    def __call__(self, hidden_states, layer_idx=None):
        # pick act
        if self.hidden_act=="relu":
            act_fn = lambda x: nn.relu(x)
        elif self.hidden_act=="gelu":
            act_fn = lambda x: nn.gelu(x)
        else:
            act_fn = lambda x: nn.relu(x)

        if self.num_codebooks<=1:
            x = nn.Dense(self.ffn_dim, use_bias=False, name="fc1")(hidden_states)
        else:
            x = MoshiFlexibleLinearFL(self.hidden_size, self.ffn_dim, self.num_codebooks, name="fc1")(
                hidden_states, layer_idx=layer_idx
            )

        B,S,F = x.shape
        x = x.reshape(B,S,2,self.ffn_dim//2)
        gate = act_fn(x[...,0,:])
        val  = x[...,1,:]
        x = gate*val

        if self.num_codebooks<=1:
            x = nn.Dense(self.hidden_size, use_bias=False, name="fc2")(x)
        else:
            x = MoshiFlexibleLinearFL(self.ffn_dim//2, self.hidden_size, self.num_codebooks, name="fc2")(
                x, layer_idx=layer_idx
            )
        return x


################################################################################
# 3) Copy PT->Flax & compare
################################################################################

def load_torch_weights_into_flax_flexiblelinear(pt_layer, flax_module, rng_key, sample_x, layer_idx=None):
    vars_ = flax_module.init(rng_key, jnp.array(sample_x.cpu().numpy()), layer_idx)
    pt_w = pt_layer.weight.detach().cpu().numpy()
    new_params = dict(vars_['params'])
    new_params['weight'] = jnp.array(pt_w)
    return {'params': new_params}

def load_torch_weights_into_flax_rmsnorm(pt_layer, flax_module, rng_key, sample_x):
    vars_ = flax_module.init(rng_key, jnp.array(sample_x.cpu().numpy()))
    pt_w = pt_layer.weight.detach().cpu().numpy()
    new_params = dict(vars_['params'])
    new_params['weight'] = jnp.array(pt_w)
    return {'params': new_params}

def load_torch_weights_into_flax_gatingmlp(pt_layer, flax_module, rng_key, sample_x, layer_idx=None):
    vars_ = flax_module.init(rng_key, jnp.array(sample_x.cpu().numpy()), layer_idx)
    new_params = dict(vars_['params'])

    if pt_layer.num_codebooks<=1:
        # fc1, fc2 => normal Linear => param: fc1/kernel, fc2/kernel
        fc1_w = pt_layer.fc1.weight.detach().cpu().numpy()  # shape (out_dim, in_dim)
        fc2_w = pt_layer.fc2.weight.detach().cpu().numpy()
        # in Flax: shape for Dense => (in_dim, out_dim)
        fc1_dict = dict(new_params['fc1'])
        fc1_dict['kernel'] = jnp.array(fc1_w.T)
        new_params['fc1'] = fc1_dict

        fc2_dict = dict(new_params['fc2'])
        fc2_dict['kernel'] = jnp.array(fc2_w.T)
        new_params['fc2'] = fc2_dict
    else:
        # fc1, fc2 => MoshiFlexibleLinear => param name = 'weight'
        fc1_w = pt_layer.fc1.weight.detach().cpu().numpy()
        fc2_w = pt_layer.fc2.weight.detach().cpu().numpy()

        fc1_dict = dict(new_params['fc1'])
        fc1_dict['weight'] = jnp.array(fc1_w)
        new_params['fc1'] = fc1_dict

        fc2_dict = dict(new_params['fc2'])
        fc2_dict['weight'] = jnp.array(fc2_w)
        new_params['fc2'] = fc2_dict

    return {'params': new_params}


def compare_outputs(pt_model, jax_module, jax_params, x_torch, layer_idx=None):
    """
    1) PyTorch forward => pt_out
    2) Convert x_torch->jax => jax_out
    3) Compute L1 diff
    """
    # PT forward
    pt_out = pt_model(x_torch, layer_idx=layer_idx).detach().cpu().numpy()

    # JAX forward
    x_jax = jnp.array(x_torch.detach().cpu().numpy())
    jax_out = jax_module.apply(jax_params, x_jax, layer_idx)

    diff = np.abs(pt_out - np.array(jax_out)).mean()
    print(f"  PyTorch vs. JAX average diff = {diff:8.6f}")
    return diff


################################################################################
# 4) Demo main
################################################################################

def main():
    print("=== Final Unified Demo: PyTorch -> Flax with optional layer_idx in all modules ===")

    # 1) FlexibleLinear
    print("\n--- MoshiFlexibleLinear demo ---")
    pt_flex = MoshiFlexibleLinearPT(8, 16, 3)
    x_pt = torch.randn(2, 3, 8)
    flax_flex = MoshiFlexibleLinearFL(8, 16, 3)
    rng = jax.random.PRNGKey(0)

    flax_vars = load_torch_weights_into_flax_flexiblelinear(pt_flex, flax_flex, rng, x_pt)
    _ = compare_outputs(pt_flex, flax_flex, flax_vars, x_pt, layer_idx=None)  # same shape => no error

    # 2) RMSNorm
    print("\n--- MoshiRMSNorm demo ---")
    pt_rms = MoshiRMSNormPT(8)
    x_pt2 = torch.randn(2, 4, 8)
    flax_rms = MoshiRMSNormFL(8)
    rng = jax.random.PRNGKey(1)

    flax_vars2 = load_torch_weights_into_flax_rmsnorm(pt_rms, flax_rms, rng, x_pt2)
    # even though layer_idx is not used, we can pass None safely
    _ = compare_outputs(pt_rms, flax_rms, flax_vars2, x_pt2, layer_idx=None)

    # 3) MLP single codebook
    print("\n--- MoshiGatingMLP single codebook ---")
    pt_mlp = MoshiGatingMLPPT(8, 16, 1, "relu")
    x_pt3 = torch.randn(2, 5, 8)
    flax_mlp = MoshiGatingMLPFL(8, 16, 1, "relu")
    rng = jax.random.PRNGKey(2)

    flax_vars3 = load_torch_weights_into_flax_gatingmlp(pt_mlp, flax_mlp, rng, x_pt3)
    _ = compare_outputs(pt_mlp, flax_mlp, flax_vars3, x_pt3)

    # 4) MLP multi codebook
    print("\n--- MoshiGatingMLP multi codebook ---")
    pt_mlp2 = MoshiGatingMLPPT(8, 16, 2, "gelu")
    x_pt4 = torch.randn(2, 2, 8)
    flax_mlp2 = MoshiGatingMLPFL(8, 16, 2, "gelu")
    rng = jax.random.PRNGKey(3)

    flax_vars4 = load_torch_weights_into_flax_gatingmlp(pt_mlp2, flax_mlp2, rng, x_pt4, layer_idx=None)
    _ = compare_outputs(pt_mlp2, flax_mlp2, flax_vars4, x_pt4, layer_idx=None)

    #print("\nAll done. No more TypeError: each module can handle layer_idx=... gracefully.")


if __name__=="__main__":
    main()
