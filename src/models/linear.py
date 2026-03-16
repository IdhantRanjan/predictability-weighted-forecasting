"""DLinear and NLinear baselines.

Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023.
"""
import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # Pad edges to preserve sequence length
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        return self.avg(torch.cat([front, x, end], dim=1).permute(0, 2, 1)).permute(0, 2, 1)


class DLinear(nn.Module):
    """Trend-seasonal decomposition with separate linear projections."""
    def __init__(self, n_features: int, seq_len: int = 336, pred_len: int = 96, individual: bool = False, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.individual = individual
        self.decomp = MovingAvg(kernel_size=25)

        if individual:
            self.linear_trend = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_features)])
            self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_features)])
        else:
            self.linear_trend = nn.Linear(seq_len, pred_len)
            self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        trend = self.decomp(x)
        seasonal = x - trend

        if self.individual:
            out = torch.zeros(x.shape[0], self.pred_len, self.n_features, device=x.device)
            for i in range(self.n_features):
                out[:, :, i] = self.linear_trend[i](trend[:, :, i]) + self.linear_seasonal[i](seasonal[:, :, i])
            return out
        else:
            t = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
            s = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            return t + s


class NLinear(nn.Module):
    """Last-value normalization with a single linear projection."""
    def __init__(self, n_features: int, seq_len: int = 336, pred_len: int = 96, individual: bool = False, **kwargs):
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        self.individual = individual

        if individual:
            self.linear = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_features)])
        else:
            self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        last_val = x[:, -1:, :]
        x_norm = x - last_val

        if self.individual:
            out = torch.zeros(x.shape[0], self.pred_len, self.n_features, device=x.device)
            for i in range(self.n_features):
                out[:, :, i] = self.linear[i](x_norm[:, :, i])
        else:
            out = self.linear(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        return out + last_val
