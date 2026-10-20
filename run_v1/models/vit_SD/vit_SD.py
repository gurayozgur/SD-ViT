# Adapted from the ViT of CVLface (https://github.com/mk-minchul/CVLface),
# MIT License, Copyright (c) 2022 Minchul Kim.
"""SD-ViT backbone: a CVLface ViT whose every block depth can serve as an exit.

Training (``model.train()``) returns the full-depth embedding, the embedding of
one intermediate block sampled uniformly from {1, ..., L-1}, and the sampled
index. Both embeddings are produced by the *same* head (LayerNorm + feature
projection), which is what places all depths in one embedding space.

Inference (``model.eval()``) runs only the first ``exit_index`` blocks and
decodes them with that same head. ``exit_index=None`` is the full-depth model.
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: Optional[float] = None,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, num_token, embed_dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, num_token, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False,
                 qk_scale: Optional[float] = None, drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 act_layer: Callable = nn.ReLU6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=112, patch_size=8, in_channels=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, height, width = x.shape
        assert height == self.img_size[0] and width == self.img_size[1], \
            f"Input image size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):

    def __init__(self, img_size: int = 112, patch_size: int = 8, in_channels: int = 3, num_classes: int = 512,
                 embed_dim: int = 512, depth: int = 12, num_heads: int = 8, mlp_ratio: float = 5.,
                 qkv_bias: bool = False, qk_scale: Optional[float] = None, drop_rate: float = 0.,
                 attn_drop_rate: float = 0., drop_path_rate: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth
        # Number of blocks run at inference. None means all of them (full depth).
        self.exit_index = None

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                                      embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)])

        # Shared head H: one LayerNorm and one projection phi for every exit.
        self.norm = nn.LayerNorm(embed_dim)
        self.feature = nn.Sequential(
            nn.Linear(in_features=embed_dim * self.num_patches, out_features=embed_dim, bias=False),
            nn.BatchNorm1d(num_features=embed_dim, eps=2e-5),
            nn.Linear(in_features=embed_dim, out_features=num_classes, bias=False),
            nn.BatchNorm1d(num_features=num_classes, eps=2e-5)
        )

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def head(self, tokens):
        """Shared head H: LayerNorm, flatten, projection phi."""
        x = self.norm(tokens.float())
        x = torch.reshape(x, (x.shape[0], self.num_patches * self.embed_dim))
        return self.feature(x)

    def embed(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_train(self, x):
        x = self.embed(x)
        # k ~ U{1, ..., L-1}, stored 0-based as block index 0, ..., L-2.
        selected_block_idx = torch.randint(0, len(self.blocks) - 1, (1,), device=x.device).item()
        secondary_tokens = None
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if block_idx == selected_block_idx:
                secondary_tokens = x
        return self.head(x), self.head(secondary_tokens), selected_block_idx

    def forward_exit(self, x, exit_index=None):
        exit_index = self.depth if exit_index is None else int(exit_index)
        if not 1 <= exit_index <= self.depth:
            raise ValueError(f'exit_index must be in [1, {self.depth}], got {exit_index}')
        x = self.embed(x)
        for block in self.blocks[:exit_index]:
            x = block(x)
        return self.head(x)

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        return self.forward_exit(x, self.exit_index)
