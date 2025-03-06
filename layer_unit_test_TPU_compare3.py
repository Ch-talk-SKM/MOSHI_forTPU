#!/usr/bin/env python3
# coding=utf-8

"""
layer_unit_test_TPU.py

(Extended to compare 3 versions of each module:
(A) *PT_origin  (from modeling_moshi.py)
(B) *PT
(C) *FL

We focus on MoshiFlexibleLinear, MoshiRMSNorm, MoshiGatingMLP.

Goal:
1) *PT_origin -> *FL weight load & compare
2) *PT -> *FL weight load & compare
3) *PT_origin -> *PT compare

*PT_origin & *FL matching is the most important, but we also check *PT vs *FL, *PT_origin vs *PT.
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
# 0) Common
################################################################################
def compare_two_model_outputs(pt_modelA, pt_modelB, x_torch, layer_idx=None, labelA="A", labelB="B"):
    """
    Compare two PyTorch model outputs.
    """
    with torch.no_grad():
        outA = pt_modelA(x_torch, layer_idx=layer_idx).detach().cpu().numpy()
        outB = pt_modelB(x_torch, layer_idx=layer_idx).detach().cpu().numpy()
    diff = np.abs(outA - outB).mean()
    print(f"  {labelA} vs {labelB} mean diff = {diff:.6e}")
    return diff

def compare_pt_fl(pt_model, fl_module, fl_params, x_torch, layer_idx=None, labelPT="", labelFL="FL"):
    """
    Compare PyTorch vs Flax outputs.
    """
    with torch.no_grad():
        out_pt = pt_model(x_torch, layer_idx=layer_idx).detach().cpu().numpy()
    x_fl = jnp.array(x_torch.detach().cpu().numpy())
    out_fl = fl_module.apply({"params": fl_params}, x_fl, layer_idx)
    out_fl_np = np.array(out_fl)
    diff = np.abs(out_pt - out_fl_np).mean()
    print(f"  {labelPT} vs {labelFL} mean diff = {diff:.6e}")
    return diff

################################################################################
# 1) Original classes from modeling_moshi.py => *_PT_origin
################################################################################

class MoshiFlexibleLinearPT_origin(pt_nn.Module):
    """
    원본 PyTorch FlexibleLinear from modeling_moshi.py
    (num_layers, out_dim, in_dim)
    """
    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__()
        self.weight = pt_nn.Parameter(torch.randn(num_layers, output_dim, input_dim))

    def forward(self, x, layer_idx=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B,S,D = x.shape
        nl, od, idim = self.weight.shape
        if D != idim:
            raise ValueError("dim mismatch in MoshiFlexibleLinearPT_origin")

        if layer_idx is not None:
            if isinstance(layer_idx, int):
                # single
                w = self.weight[layer_idx]
                return F.linear(x, w)
            elif layer_idx.dim()==1:
                L = layer_idx.shape[0]
                if S==L:
                    outs=[]
                    for i in range(L):
                        outs.append(F.linear(x[:,i], self.weight[layer_idx[i]]))
                    return torch.stack(outs,dim=1)
                if S==1:
                    x_squeezed = x[:,0]
                    outs=[]
                    for idx in layer_idx:
                        outs.append(F.linear(x_squeezed, self.weight[idx]))
                    return torch.stack(outs,dim=1)
                else:
                    raise ValueError("shape mismatch in PT_origin flexible usage")
            else:
                raise ValueError("layer_idx must be int or 1D tensor for PT_origin usage")
        else:
            if S!=nl:
                raise ValueError("S!=num_layers in PT_origin flexible usage")
            outs=[]
            for i in range(S):
                w_i = self.weight[i]
                outs.append(F.linear(x[:,i], w_i))
            return torch.stack(outs,dim=1)

class MoshiRMSNormPT_origin(pt_nn.Module):
    """
    원본 PyTorch RMSNorm from modeling_moshi.py
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = pt_nn.Parameter(torch.ones(dim))

    def forward(self, x, layer_idx=None):
        # ignore layer_idx
        x_f32 = x.float()
        rms = torch.sqrt((x_f32**2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x_f32 / rms
        scaled = normed * self.weight.float()
        return scaled.type_as(x)

class MoshiGatingMLPPT_origin(pt_nn.Module):
    """
    원본 Gating MLP from modeling_moshi.py
    """
    def __init__(self, hidden_size, ffn_dim, num_codebooks=1, hidden_act="relu"):
        super().__init__()
        if ffn_dim<2 or ffn_dim%2!=0:
            raise ValueError("ffn_dim must be even >=2")

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_codebooks = num_codebooks

        if hidden_act=="relu":
            self.act_fn = pt_nn.ReLU()
        elif hidden_act=="gelu":
            self.act_fn = pt_nn.GELU()
        else:
            self.act_fn = pt_nn.ReLU()

        if num_codebooks<=1:
            self.fc1 = pt_nn.Linear(hidden_size, ffn_dim, bias=False)
            self.fc2 = pt_nn.Linear(ffn_dim//2, hidden_size, bias=False)
        else:
            self.fc1 = MoshiFlexibleLinearPT_origin(hidden_size, ffn_dim, num_codebooks)
            self.fc2 = MoshiFlexibleLinearPT_origin(ffn_dim//2, hidden_size, num_codebooks)

    def forward(self, x, layer_idx=None):
        if self.num_codebooks<=1:
            x = self.fc1(x)
        else:
            x = self.fc1(x, layer_idx=layer_idx)

        B,S,F = x.shape
        x = x.view(B,S,2,-1)
        gate = self.act_fn(x[...,0,:])
        val  = x[...,1,:]
        x = gate*val

        if self.num_codebooks<=1:
            x = self.fc2(x)
        else:
            x = self.fc2(x, layer_idx=layer_idx)
        return x

################################################################################
# 2) PyTorch modules => *_PT
################################################################################

class MoshiFlexibleLinearPT(pt_nn.Module):
    """
    Flexible linear (PyTorch) with optional layer_idx.
    """
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >=1.")
        self.num_layers = num_layers
        self.weight = pt_nn.Parameter(torch.randn(num_layers, output_size, input_size))

    def forward(self, x, layer_idx=None):
        if x.dim()==2:
            x = x.unsqueeze(1)
        B,S,D = x.shape
        nl, od, idim = self.weight.shape
        if D!=idim:
            raise ValueError("dim mismatch in MoshiFlexibleLinearPT")

        if layer_idx is None:
            if S!=nl:
                raise ValueError("S!=num_layers in PT flexible usage")
            outs=[]
            for i in range(S):
                w_i = self.weight[i]
                outs.append(F.linear(x[:,i], w_i))
            return torch.stack(outs, dim=1)

        if isinstance(layer_idx,int):
            if not(0<=layer_idx<nl):
                raise ValueError("invalid layer_idx in PT flexible linear usage.")
            w = self.weight[layer_idx]
            return F.linear(x, w)

        if layer_idx.dim()==1:
            L = layer_idx.shape[0]
            if S==L:
                outs=[]
                for i in range(L):
                    outs.append(F.linear(x[:,i], self.weight[layer_idx[i]]))
                return torch.stack(outs, dim=1)
            if S==1:
                x_squeezed = x[:,0]
                outs=[]
                for idx in layer_idx:
                    outs.append(F.linear(x_squeezed, self.weight[idx]))
                return torch.stack(outs, dim=1)
            raise ValueError("shape mismatch in PT flexible linear usage.")
        raise ValueError("layer_idx must be None/int/1D in PT flexible usage")

class MoshiRMSNormPT(pt_nn.Module):
    """
    PyTorch RMSNorm with optional layer_idx
    ignoring layer_idx
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = pt_nn.Parameter(torch.ones(dim))

    def forward(self, x, layer_idx=None):
        x_f32 = x.float()
        rms = torch.sqrt((x_f32**2).mean(dim=-1, keepdim=True)+self.eps)
        normed = x_f32/rms
        scaled = normed*self.weight.float()
        return scaled.type_as(x)


_ACT2FN = {
    "relu": pt_nn.ReLU(),
    "gelu": pt_nn.GELU(),
}

class MoshiGatingMLPPT(pt_nn.Module):
    """
    Gating MLP in PyTorch with optional layer_idx
    """
    def __init__(self, hidden_size, ffn_dim, num_codebooks=1, hidden_act="relu"):
        super().__init__()
        if ffn_dim<2 or ffn_dim%2!=0:
            raise ValueError("ffn_dim must be even >=2")
        self.hidden_act = hidden_act
        self.activation_fn = _ACT2FN.get(hidden_act, pt_nn.ReLU())
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_codebooks=num_codebooks

        if num_codebooks<=1:
            self.fc1 = pt_nn.Linear(hidden_size, ffn_dim, bias=False)
            self.fc2 = pt_nn.Linear(ffn_dim//2, hidden_size, bias=False)
        else:
            self.fc1 = MoshiFlexibleLinearPT(hidden_size, ffn_dim, num_codebooks)
            self.fc2 = MoshiFlexibleLinearPT(ffn_dim//2, hidden_size, num_codebooks)

    def forward(self, x, layer_idx=None):
        if self.num_codebooks<=1:
            x = self.fc1(x)
        else:
            x = self.fc1(x, layer_idx=layer_idx)

        B,S,F = x.shape
        x = x.view(B,S,2,-1)
        gate = self.activation_fn(x[...,0,:])
        val  = x[...,1,:]
        x = gate*val

        if self.num_codebooks<=1:
            x = self.fc2(x)
        else:
            x = self.fc2(x, layer_idx=layer_idx)
        return x


################################################################################
# 3) Flax modules => *_FL
################################################################################

class MoshiFlexibleLinearFL(nn.Module):
    input_size: int
    output_size: int
    num_layers: int

    @nn.compact
    def __call__(self, x, layer_idx=None):
        weight = self.param("weight", nn.initializers.normal(stddev=1.0),
                            (self.num_layers, self.output_size, self.input_size))
        if x.ndim==2:
            x = x[:,None,:]
        B,S,D = x.shape
        if D!=self.input_size:
            raise ValueError("dimension mismatch in Flax flexible linear")

        def matmul_2d(v,w):
            return jnp.einsum("bd,od->bo", v,w)

        if layer_idx is None:
            if S!=self.num_layers:
                raise ValueError("S!=num_layers in Flax flexible linear usage.")
            outs=[]
            for i in range(S):
                w_i = weight[i]
                outs.append(matmul_2d(x[:,i], w_i))
            return jnp.stack(outs, axis=1)
        if isinstance(layer_idx,int):
            if not (0<= layer_idx < self.num_layers):
                raise ValueError("invalid layer_idx in flax flexible usage")
            w = weight[layer_idx]
            outs=[]
            for i in range(S):
                outs.append(matmul_2d(x[:,i], w))
            return jnp.stack(outs,axis=1)

        if layer_idx is not None and layer_idx.ndim==1:
            L = layer_idx.shape[0]
            if S==L:
                outs=[]
                for i in range(L):
                    outs.append(matmul_2d(x[:,i], weight[layer_idx[i]]))
                return jnp.stack(outs,axis=1)
            if S==1:
                x_squeezed= x[:,0]
                outs=[]
                for idx in layer_idx:
                    outs.append(matmul_2d(x_squeezed, weight[idx]))
                return jnp.stack(outs,axis=1)
            raise ValueError("shape mismatch in Flax flexible usage.")
        raise ValueError("layer_idx must be None/int/1D in Flax flexible usage")

class MoshiRMSNormFL(nn.Module):
    dim: int
    eps: float=1e-6

    @nn.compact
    def __call__(self, x, layer_idx=None):
        weight = self.param("weight", nn.initializers.ones,(self.dim,))
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True)+self.eps)
        normed = x_f32/rms
        scaled = normed*weight
        return scaled.astype(x.dtype)


class MoshiGatingMLPFL(nn.Module):
    hidden_size: int
    ffn_dim: int
    num_codebooks:int=1
    hidden_act:str="relu"

    @nn.compact
    def __call__(self, hidden_states, layer_idx=None):
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
# 4) Load PT->Flax
################################################################################

def load_pt_weights_flexlinear(pt_layer, fl_layer, rng, sample_input, layer_idx=None):
    """
    (PT_origin or PT) => FL
    """
    init_vars = fl_layer.init(rng, jnp.array(sample_input.detach().cpu().numpy()), layer_idx)
    new_params = dict(init_vars["params"])
    pt_w = pt_layer.weight.detach().cpu().numpy()
    new_params["weight"] = jnp.array(pt_w)
    return {"params": new_params}

def load_pt_weights_rmsnorm(pt_layer, fl_layer, rng, sample_input):
    init_vars = fl_layer.init(rng, jnp.array(sample_input.detach().cpu().numpy()))
    new_params = dict(init_vars["params"])
    pt_w = pt_layer.weight.detach().cpu().numpy()
    new_params["weight"] = jnp.array(pt_w)
    return {"params": new_params}

def load_pt_weights_gatingmlp(pt_layer, fl_layer, rng, sample_input, layer_idx=None):
    init_vars = fl_layer.init(rng, jnp.array(sample_input.detach().cpu().numpy()), layer_idx)
    new_params = dict(init_vars["params"])

    # PT => if num_codebooks<=1 => fc1= Linear => param: kernel
    # else => flexible => param: weight
    if pt_layer.num_codebooks<=1:
        fc1_w = pt_layer.fc1.weight.detach().cpu().numpy()  # shape (out_dim, in_dim)
        fc2_w = pt_layer.fc2.weight.detach().cpu().numpy()
        # Flax => fc1/kernel => (in_dim,out_dim)
        fc1_dict = dict(new_params["fc1"])
        fc1_dict["kernel"] = jnp.array(fc1_w.T)
        new_params["fc1"] = fc1_dict

        fc2_dict = dict(new_params["fc2"])
        fc2_dict["kernel"] = jnp.array(fc2_w.T)
        new_params["fc2"] = fc2_dict
    else:
        fc1_w = pt_layer.fc1.weight.detach().cpu().numpy()
        fc2_w = pt_layer.fc2.weight.detach().cpu().numpy()
        fc1_dict = dict(new_params["fc1"])
        fc1_dict["weight"] = jnp.array(fc1_w)
        new_params["fc1"] = fc1_dict

        fc2_dict = dict(new_params["fc2"])
        fc2_dict["weight"] = jnp.array(fc2_w)
        new_params["fc2"] = fc2_dict

    return {"params": new_params}

################################################################################
# 5) Demo main
################################################################################

def main():
    print("=== layer_unit_test_TPU.py: with 3-way compare: (PT_origin), (PT), (FL) ===")

    # 1) FlexibleLinear
    print("\n--- Compare FlexibleLinear 3 ways ---")
    # (A) PT_origin
    pt_flex_origin = MoshiFlexibleLinearPT_origin(8,16,3)
    # (B) PT_test
    pt_flex_test = MoshiFlexibleLinearPT(8,16,3)
    # (C) FL
    fl_flex = MoshiFlexibleLinearFL(8,16,3)

    x_pt = torch.randn(2,3,8)
    rng = jax.random.PRNGKey(0)

    # PT_origin->FL
    fl_flex_varsA = load_pt_weights_flexlinear(pt_flex_origin, fl_flex, rng, x_pt)
    # PT_test->FL
    rng2 = jax.random.PRNGKey(1)
    fl_flex_varsB = load_pt_weights_flexlinear(pt_flex_test, fl_flex, rng2, x_pt)

    print("Compare (PT_origin) vs (PT_test):")
    compare_two_model_outputs(pt_flex_origin, pt_flex_test, x_pt, labelA="PT_origin", labelB="PT_test")

    print("Compare (PT_origin) vs (FL): (loaded from PT_origin)")
    compare_pt_fl(pt_flex_origin, fl_flex, fl_flex_varsA["params"], x_pt, labelPT="PT_origin")

    print("Compare (PT_test)   vs (FL): (loaded from PT_test)")
    compare_pt_fl(pt_flex_test, fl_flex, fl_flex_varsB["params"], x_pt, labelPT="PT_test")


    # 2) RMSNorm
    print("\n--- Compare RMSNorm 3 ways ---")
    pt_rms_origin = MoshiRMSNormPT_origin(8)
    pt_rms_test = MoshiRMSNormPT(8)
    fl_rms = MoshiRMSNormFL(8)
    x_pt2 = torch.randn(2,4,8)

    rng3 = jax.random.PRNGKey(2)
    fl_rms_varsA = load_pt_weights_rmsnorm(pt_rms_origin, fl_rms, rng3, x_pt2)
    rng4 = jax.random.PRNGKey(3)
    fl_rms_varsB = load_pt_weights_rmsnorm(pt_rms_test, fl_rms, rng4, x_pt2)

    print("Compare (PT_origin) vs (PT_test):")
    compare_two_model_outputs(pt_rms_origin, pt_rms_test, x_pt2, labelA="PT_origin", labelB="PT_test")

    print("Compare (PT_origin) vs (FL): (loaded from PT_origin)")
    compare_pt_fl(pt_rms_origin, fl_rms, fl_rms_varsA["params"], x_pt2, labelPT="PT_origin")
    print("Compare (PT_test) vs (FL): (loaded from PT_test)")
    compare_pt_fl(pt_rms_test, fl_rms, fl_rms_varsB["params"], x_pt2, labelPT="PT_test")


    # 3) MLP single codebook
    print("\n--- Compare MLP single codebook 3 ways ---")
    pt_mlp_origin = MoshiGatingMLPPT_origin(8,16,1,"relu")
    pt_mlp_test = MoshiGatingMLPPT(8,16,1,"relu")
    fl_mlp = MoshiGatingMLPFL(8,16,1,"relu")

    x_pt3 = torch.randn(2,5,8)
    rng5 = jax.random.PRNGKey(4)
    fl_mlp_varsA = load_pt_weights_gatingmlp(pt_mlp_origin, fl_mlp, rng5, x_pt3)
    rng6 = jax.random.PRNGKey(5)
    fl_mlp_varsB = load_pt_weights_gatingmlp(pt_mlp_test, fl_mlp, rng6, x_pt3)

    print("Compare (PT_origin) vs (PT_test):")
    compare_two_model_outputs(pt_mlp_origin, pt_mlp_test, x_pt3, labelA="PT_origin", labelB="PT_test")

    print("Compare (PT_origin) vs (FL): (loaded from PT_origin)")
    compare_pt_fl(pt_mlp_origin, fl_mlp, fl_mlp_varsA["params"], x_pt3, labelPT="PT_origin")
    print("Compare (PT_test) vs (FL): (loaded from PT_test)")
    compare_pt_fl(pt_mlp_test, fl_mlp, fl_mlp_varsB["params"], x_pt3, labelPT="PT_test")


    # 4) MLP multi codebook
    print("\n--- Compare MLP multi codebook 3 ways ---")
    pt_mlp2_origin = MoshiGatingMLPPT_origin(8,16,2,"gelu")
    pt_mlp2_test = MoshiGatingMLPPT(8,16,2,"gelu")
    fl_mlp2 = MoshiGatingMLPFL(8,16,2,"gelu")

    x_pt4 = torch.randn(2,2,8)
    rng7 = jax.random.PRNGKey(6)
    fl_mlp2_varsA = load_pt_weights_gatingmlp(pt_mlp2_origin, fl_mlp2, rng7, x_pt4)
    rng8 = jax.random.PRNGKey(7)
    fl_mlp2_varsB = load_pt_weights_gatingmlp(pt_mlp2_test, fl_mlp2, rng8, x_pt4)

    print("Compare (PT_origin) vs (PT_test):")
    compare_two_model_outputs(pt_mlp2_origin, pt_mlp2_test, x_pt4, labelA="PT_origin", labelB="PT_test")

    print("Compare (PT_origin) vs (FL): (loaded from PT_origin)")
    compare_pt_fl(pt_mlp2_origin, fl_mlp2, fl_mlp2_varsA["params"], x_pt4, labelPT="PT_origin")
    print("Compare (PT_test) vs (FL): (loaded from PT_test)")
    compare_pt_fl(pt_mlp2_test, fl_mlp2, fl_mlp2_varsB["params"], x_pt4, labelPT="PT_test")

    print("\nAll done. If (PT_origin vs FL) diffs are small => success.")


if __name__=="__main__":
    main()
