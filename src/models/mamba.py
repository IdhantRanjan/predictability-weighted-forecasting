"""Mamba-inspired selective SSM for time-series forecasting.

Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023.
This implements the key ideas (input-dependent gating, causal convolution) using
standard PyTorch ops rather than the custom CUDA kernels from the original paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveGatedConv(nn.Module):
    """Core selective state space block.

    The xz split + causal conv + input-dependent gate mirrors Mamba's design;
    we skip the parallel scan since for our sequence lengths the conv approximation
    has similar capacity with much simpler code.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1)
        self.gate_proj = nn.Linear(self.d_inner, self.d_inner)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        x_branch, z = self.in_proj(x).chunk(2, dim=-1)
        h = self.conv(x_branch.transpose(1, 2))[..., :L].transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x_branch))
        y = F.silu(h * gate) * F.silu(z) + self.D * x_branch
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveGatedConv(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.ssm(self.norm(x)))


class MambaModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int = 336,
        pred_len: int = 96,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.feature_proj = nn.Linear(d_model, n_features)
        self.temporal_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        last_val = x[:, -1:, :]
        h = self.input_proj(x - last_val)
        for block in self.blocks:
            h = block(h)
        h = self.feature_proj(self.norm(h))
        out = self.temporal_proj(h.permute(0, 2, 1)).permute(0, 2, 1)
        return out + last_val
