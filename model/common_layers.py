from typing import Any, Callable, Optional, cast
from numpy import isin
import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath


class MLP(nn.Module):
    """
    multi-layer perceptron
    """

    fc1: nn.Linear
    act: nn.Module
    fc2: nn.Linear
    drop: nn.Dropout

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    num_heads: int
    scale: float
    qkv: nn.Linear
    attn_drop: nn.Dropout
    proj: nn.Linear
    proj_drop: nn.Dropout
    edge_embedding: nn.Linear

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.edge_embedding = nn.Linear(17 * 17, 17 * 17)

    def forward(self, x: Tensor, edge_embedding: Tensor):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        edge_embedding = self.edge_embedding(edge_embedding)
        edge_embedding = (
            edge_embedding.reshape(1, 17, 17)
            .unsqueeze(0)
            .repeat(B, self.num_heads, 1, 1)
        )

        attn = attn + edge_embedding

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CvaAttention(nn.Module):
    num_heads: int
    scale: float
    q_norm: nn.LayerNorm
    k_norm: nn.LayerNorm
    v_norm: nn.LayerNorm
    q_linear: nn.Linear
    k_linear: nn.Linear
    v_linear: nn.Linear
    qkv: nn.Linear
    attn_drop: nn.Dropout
    proj: nn.Linear
    proj_drop: nn.Dropout
    edge_embedding: nn.Linear

    def __init__(
        self,
        dim: int,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.edge_embedding = nn.Linear(17 * 17, 17 * 17)

    def forward(
        self,
        x: Tensor,
        cva_input: Tensor,
        edge_embedding: Tensor,
    ):
        B, N, C = x.shape
        # CVA_input = self.max_pool(CVA_input)
        # print(CVA_input.shape)
        q = (
            self.q_linear(self.q_norm(cva_input))
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_linear(self.k_norm(cva_input))
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_linear(self.v_norm(x))
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale

        edge_embedding = self.edge_embedding(edge_embedding)
        edge_embedding = (
            edge_embedding.reshape(1, 17, 17)
            .unsqueeze(0)
            .repeat(B, self.num_heads, 1, 1)
        )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    norm1: nn.Module
    attn: Attention
    drop_path: nn.Identity | DropPath
    norm2: nn.Module
    mlp: MLP

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: Tensor, edge_embedding: Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x), edge_embedding))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiOutBlock(nn.Module):
    norm1: nn.Module
    attn: Attention
    drop_path: nn.Module
    norm2: nn.Module
    mlp: MLP
    norm_hop1: nn.Module
    norm_hop2: nn.Module
    mlp_hop: MLP

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.norm_hop1 = norm_layer(dim)
        self.norm_hop2 = norm_layer(dim)
        self.mlp_hop = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: Tensor, hops: Tensor, edge_embedding: Tensor):
        MSA: Tensor = self.drop_path(self.attn(self.norm1(x), edge_embedding))
        MSA = self.norm_hop1(hops) * MSA

        x = x + MSA
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        hops = hops + MSA
        hops = hops + self.drop_path(self.mlp_hop(self.norm_hop2(hops)))

        return x, hops, MSA


class MultiInOutBlock(nn.Module):
    norm1: nn.Module
    attn: Attention
    drop_path: nn.Module
    norm2: nn.Module
    mlp: MLP
    norm_hop1: nn.Module
    norm_hop2: nn.Module
    mlp_hop: MLP

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale: Optional[float] = None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.cva_attn = CvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # self.max_pool = nn.MaxPool1d(3, stride=1, padding=1, dilation=1, return_indices=False, ceil_mode=False)

        self.norm_hop1 = norm_layer(dim)
        self.norm_hop2 = norm_layer(dim)
        self.mlp_hop = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self, x: Tensor, hops: Tensor, CVA_input: Tensor, edge_embedding: Tensor
    ):
        MSA: Tensor = self.drop_path(self.cva_attn(x, CVA_input, edge_embedding))
        MSA = self.norm_hop1(hops) * MSA

        x = x + MSA
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        hops = hops + MSA
        hops = hops + self.drop_path(self.mlp_hop(self.norm_hop2(hops)))
        return x, hops, MSA

