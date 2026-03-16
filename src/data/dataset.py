"""Sliding-window dataset for multivariate time-series forecasting."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """Each sample: (x, y, pred_score) where x is input, y is target, and
    pred_score is the local predictability score for the target window.
    """
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 336,
        pred_len: int = 96,
        split: str = "train",
        predictability_scores: np.ndarray = None,
        scaler: StandardScaler = None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len

        n = len(data)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)

        if split == "train":
            raw = data[:train_end]
        elif split == "val":
            raw = data[train_end - seq_len : val_end]
        else:
            raw = data[val_end - seq_len :]

        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(data[:train_end])
        else:
            self.scaler = scaler

        self.data = self.scaler.transform(raw).astype(np.float32)

        if predictability_scores is not None:
            if split == "train":
                self.pred_scores = predictability_scores[:train_end]
            elif split == "val":
                self.pred_scores = predictability_scores[train_end - seq_len : val_end]
            else:
                self.pred_scores = predictability_scores[val_end - seq_len :]
        else:
            self.pred_scores = np.ones(len(self.data), dtype=np.float32)

        self.n_samples = max(0, len(self.data) - seq_len - pred_len + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        pred_score = float(np.mean(self.pred_scores[idx + self.seq_len : idx + self.seq_len + self.pred_len]))
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(pred_score, dtype=torch.float32)

    def get_scaler(self):
        return self.scaler


def create_datasets(df: pd.DataFrame, seq_len: int = 336, pred_len: int = 96, predictability_scores: np.ndarray = None):
    data = df.select_dtypes(include=[np.number]).values
    train_ds = TimeSeriesDataset(data, seq_len, pred_len, "train", predictability_scores=predictability_scores)
    val_ds = TimeSeriesDataset(data, seq_len, pred_len, "val", predictability_scores=predictability_scores, scaler=train_ds.get_scaler())
    test_ds = TimeSeriesDataset(data, seq_len, pred_len, "test", predictability_scores=predictability_scores, scaler=train_ds.get_scaler())
    return train_ds, val_ds, test_ds
