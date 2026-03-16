"""PatchTST for time-series forecasting.

Nie et al., "A Time Series is Worth 64 Words", ICLR 2023.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, n_features: int, patch_len: int = 16, stride: int = 8, d_model: int = 128):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x):
        B, L, F = x.shape
        n_patches = (L - self.patch_len) // self.stride + 1
        patches = torch.stack(
            [x[:, i * self.stride : i * self.stride + self.patch_len, :].reshape(B, -1)
             for i in range(n_patches)],
            dim=1
        )
        return self.proj(patches)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        return x + self.ff(self.norm2(x))


class PatchTST(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int = 336,
        pred_len: int = 96,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        self.patch_embed = PatchEmbedding(n_features, patch_len, stride, d_model)
        n_patches = (seq_len - patch_len) // stride + 1
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model * n_patches, pred_len * n_features)

    def forward(self, x):
        B = x.shape[0]
        last_val = x[:, -1:, :]
        h = self.patch_embed(x - last_val) + self.pos_embed
        for block in self.blocks:
            h = block(h)
        out = self.head(self.norm(h).reshape(B, -1)).reshape(B, self.pred_len, self.n_features)
        return out + last_val
