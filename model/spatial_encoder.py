from typing import Any, Callable, Optional, cast
from numpy import isin
import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath
from model.common_layers import MultiInOutBlock, MultiOutBlock

## Our model was revised from https://github.com/zczcwh/PoseFormer/blob/main/common/model_poseformer.py
class FirstViewSpatialFeatures(nn.Module):
    spatial_patch_to_embedding: nn.Linear
    spatial_pos_embed: nn.Parameter
    spatial_norm: nn.Module
    hop_to_embedding: nn.Linear
    hop_pos_embed: nn.Parameter
    pos_drop: nn.Dropout
    hop_norm: nn.Module
    block1: MultiOutBlock
    block2: MultiOutBlock
    block3: MultiOutBlock
    block4: MultiOutBlock

    def __init__(
        self,
        num_frame=9,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=32,
        depth=4,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale: Optional[float] = None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        ### spatial patch embedding
        self.spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_joints, embed_dim_ratio)
        )

        self.hop_to_embedding = nn.Linear(68, embed_dim_ratio)
        self.hop_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.block1 = MultiOutBlock(
            dim=embed_dim_ratio,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[0],
            norm_layer=norm_layer,
        )
        self.block2 = MultiOutBlock(
            dim=embed_dim_ratio,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[1],
            norm_layer=norm_layer,
        )
        self.block3 = MultiOutBlock(
            dim=embed_dim_ratio,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[2],
            norm_layer=norm_layer,
        )
        self.block4 = MultiOutBlock(
            dim=embed_dim_ratio,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[3],
            norm_layer=norm_layer,
        )

        self.spatial_norm = norm_layer(embed_dim_ratio)

        self.hop_norm = norm_layer(embed_dim_ratio)

    def forward(
        self,
        x: Tensor,
        hops: Tensor,
        edge_embedding: Tensor,
    ):
        b, _, f, p = (
            x.shape
        )  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(
            x,
            "b c f p  -> (b f) p  c",
        )

        x = self.spatial_patch_to_embedding(x)
        x += self.spatial_pos_embed
        x = self.pos_drop(x)

        hops = rearrange(
            hops,
            "b c f p  -> (b f) p  c",
        )
        hops = self.hop_to_embedding(hops)
        hops += self.hop_pos_embed
        hops = self.pos_drop(hops)

        x, hops, MSA1 = self.block1(x, hops, edge_embedding)
        x, hops, MSA2 = self.block2(x, hops, edge_embedding)
        x, hops, MSA3 = self.block3(x, hops, edge_embedding)
        x, hops, MSA4 = self.block4(x, hops, edge_embedding)
        MSA1 = cast(Tensor, MSA1)
        MSA2 = cast(Tensor, MSA2)
        MSA3 = cast(Tensor, MSA3)
        MSA4 = cast(Tensor, MSA4)

        x = self.spatial_norm(x)
        x = rearrange(x, "(b f) w c -> b f (w c)", f=f)

        hops = self.hop_norm(hops)
        hops = rearrange(hops, "(b f) w c -> b f (w c)", f=f)

        return x, hops, MSA1, MSA2, MSA3, MSA4


class SpatialFeatures(nn.Module):
    spatial_patch_to_embedding: nn.Linear
    spatial_pos_embed: nn.Parameter
    pos_drop: nn.Dropout
    hop_to_embedding: nn.Linear
    hop_pos_embed: nn.Parameter
    block1: MultiInOutBlock
    block2: MultiInOutBlock
    block3: MultiInOutBlock
    block4: MultiInOutBlock
    spatial_norm: nn.Module
    hop_norm: nn.Module

    def __init__(
        self,
        num_frame=9,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=32,
        depth=4,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=None,
    ):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        ### spatial patch embedding
        self.spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_joints, embed_dim_ratio)
        )

        self.hop_to_embedding = nn.Linear(68, embed_dim_ratio)
        self.hop_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.block1 = MultiInOutBlock(
            dim=embed_dim_ratio,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[0],
            norm_layer=norm_layer,
        )
        self.block2 = MultiInOutBlock(
            dim=embed_dim_ratio,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[1],
            norm_layer=norm_layer,
        )
        self.block3 = MultiInOutBlock(
            dim=embed_dim_ratio,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[2],
            norm_layer=norm_layer,
        )
        self.block4 = MultiInOutBlock(
            dim=embed_dim_ratio,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[3],
            norm_layer=norm_layer,
        )

        self.spatial_norm = norm_layer(embed_dim_ratio)

        self.hop_norm = norm_layer(embed_dim_ratio)

    def forward(
        self,
        x: Tensor,
        hops: Tensor,
        MSA1: Tensor,
        MSA2: Tensor,
        MSA3: Tensor,
        MSA4: Tensor,
        edge_embedding: Tensor,
    ):
        b, _, f, p = (
            x.shape
        )  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(
            x,
            "b c f p  -> (b f) p  c",
        )

        x = self.spatial_patch_to_embedding(x)
        x += self.spatial_pos_embed
        x = self.pos_drop(x)

        hops = rearrange(
            hops,
            "b c f p  -> (b f) p  c",
        )
        hops = self.hop_to_embedding(hops)
        hops += self.hop_pos_embed
        hops = self.pos_drop(hops)

        x, hops, MSA1 = self.block1(x, hops, MSA1, edge_embedding)
        x, hops, MSA2 = self.block2(x, hops, MSA2, edge_embedding)
        x, hops, MSA3 = self.block3(x, hops, MSA3, edge_embedding)
        x, hops, MSA4 = self.block4(x, hops, MSA4, edge_embedding)

        x = self.spatial_norm(x)
        x = rearrange(x, "(b f) w c -> b f (w c)", f=f)

        hops = self.hop_norm(hops)
        hops = rearrange(hops, "(b f) w c -> b f (w c)", f=f)

        return x, hops, MSA1, MSA2, MSA3, MSA4
