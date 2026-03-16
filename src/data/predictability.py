"""Local predictability scoring for time-series windows.

Predictability = 1 - entropy, computed via spectral or permutation entropy.
Scores are used both for stratified evaluation and for PWT loss weighting.
"""
import numpy as np
from scipy import signal
from scipy.stats import entropy


def spectral_entropy(x: np.ndarray, sf: float = 1.0, normalize: bool = True) -> float:
    freqs, psd = signal.welch(x, fs=sf, nperseg=min(len(x), 256))
    psd_norm = psd / (psd.sum() + 1e-10)
    se = entropy(psd_norm)
    if normalize:
        se = se / (np.log(len(psd_norm)) + 1e-10)
    return se


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1, normalize: bool = True) -> float:
    n = len(x)
    if n < order * delay:
        return 1.0

    from collections import Counter
    patterns = [
        tuple(np.argsort([x[j] for j in range(i, i + order * delay, delay)]))
        for i in range(n - (order - 1) * delay)
    ]
    counts = Counter(patterns)
    probs = np.array([c / len(patterns) for c in counts.values()])
    pe = entropy(probs)
    if normalize:
        pe = pe / (np.log(np.math.factorial(order)) + 1e-10)
    return pe


def compute_predictability_scores(
    data: np.ndarray,
    window_size: int = 48,
    stride: int = 24,
    method: str = "spectral",
) -> np.ndarray:
    """Per-timestep predictability scores via sliding-window entropy.

    Args:
        data: (T, D) multivariate time series
        window_size: sliding window length
        stride: step between windows
        method: 'spectral', 'permutation', or 'both'

    Returns:
        (T,) scores in [0, 1], higher = more predictable
    """
    T = len(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    D = data.shape[1]

    window_starts = list(range(0, T - window_size + 1, stride))
    window_scores = np.zeros(len(window_starts), dtype=np.float32)
    window_centers = np.zeros(len(window_starts), dtype=np.float32)

    for wi, start in enumerate(window_starts):
        end = start + window_size
        window = data[start:end]
        window_centers[wi] = (start + end) / 2.0

        entropies = []
        for d in range(D):
            col = window[:, d]
            if np.std(col) < 1e-8:
                entropies.append(0.0)
                continue
            if method == "spectral":
                e = spectral_entropy(col)
            elif method == "permutation":
                e = permutation_entropy(col)
            else:
                e = 0.5 * spectral_entropy(col) + 0.5 * permutation_entropy(col)
            entropies.append(e)

        window_scores[wi] = 1.0 - np.clip(np.mean(entropies), 0, 1)

    timesteps = np.arange(T, dtype=np.float32)
    return np.interp(timesteps, window_centers, window_scores).astype(np.float32)


def stratify_by_predictability(scores: np.ndarray, n_quartiles: int = 4) -> list:
    """Return index arrays split into n_quartiles predictability bins (low to high)."""
    thresholds = np.percentile(scores, np.linspace(0, 100, n_quartiles + 1))
    quartiles = []
    for i in range(n_quartiles):
        lo, hi = thresholds[i], thresholds[i + 1]
        mask = (scores >= lo) & (scores <= hi if i == n_quartiles - 1 else scores < hi)
        quartiles.append(np.where(mask)[0])
    return quartiles
