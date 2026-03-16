"""S4D (diagonal S4) for multivariate time-series forecasting.

Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces", ICLR 2022.
We use the diagonal parameterization from S4D which is simpler and nearly as
effective as the original HiPPO construction.
"""
import math
import torch
import torch.nn as nn
from einops import rearrange


class S4DKernel(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # HiPPO-inspired diagonal initialization: A_n = -1/2 + i*pi*n
        real = -0.5 * torch.ones(d_state)
        imag = math.pi * torch.arange(d_state).float()
        A = torch.complex(real, imag)

        self.A_log = nn.Parameter(torch.log(-A.real))
        self.A_imag = nn.Parameter(A.imag)
        self.B_re = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.B_im = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

    def _get_A(self):
        return -torch.exp(self.A_log) + 1j * self.A_imag

    def forward(self, L: int):
        """Returns (d_model, L) convolution kernel."""
        A = self._get_A()
        B = self.B_re + 1j * self.B_im
        C = self.C_re + 1j * self.C_im
        A_bar = torch.exp(A)
        CB = C * B
        arange = torch.arange(L, device=A.device, dtype=torch.float32)
        vandermonde = torch.exp(A_bar.unsqueeze(1).log() * arange.unsqueeze(0))
        return (CB @ vandermonde).real * 2


class S4Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.kernel = S4DKernel(d_model, d_state)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        L = x.shape[1]

        kernel = self.kernel(L)
        x_fft = torch.fft.rfft(x.transpose(1, 2), n=2 * L)
        k_fft = torch.fft.rfft(kernel, n=2 * L)
        y = torch.fft.irfft(x_fft * k_fft, n=2 * L)[..., :L]
        y = y + self.kernel.D.unsqueeze(0).unsqueeze(-1) * x.transpose(1, 2)
        x = residual + self.dropout1(y.transpose(1, 2))
        return x + self.ff(self.norm2(x))


class S4Model(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int = 336,
        pred_len: int = 96,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([S4Block(d_model, d_state, dropout) for _ in range(n_layers)])
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
