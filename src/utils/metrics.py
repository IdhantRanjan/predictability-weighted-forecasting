"""Evaluation metrics and error decomposition."""
import numpy as np
import torch


def mse(pred, target):
    if isinstance(pred, torch.Tensor):
        return ((pred - target) ** 2).mean().item()
    return np.mean((pred - target) ** 2)


def mae(pred, target):
    if isinstance(pred, torch.Tensor):
        return (pred - target).abs().mean().item()
    return np.mean(np.abs(pred - target))


def stratified_metrics(preds: np.ndarray, targets: np.ndarray, pred_scores: np.ndarray, n_quartiles: int = 4) -> dict:
    """MSE/MAE broken down by predictability quartile."""
    results = {"overall_mse": mse(preds, targets), "overall_mae": mae(preds, targets), "quartiles": []}
    thresholds = np.percentile(pred_scores, np.linspace(0, 100, n_quartiles + 1))

    for i in range(n_quartiles):
        lo, hi = thresholds[i], thresholds[i + 1]
        mask = (pred_scores >= lo) & (pred_scores <= hi if i == n_quartiles - 1 else pred_scores < hi)
        if mask.sum() == 0:
            results["quartiles"].append({"quartile": i + 1, "n_samples": 0, "avg_predictability": 0, "mse": float("nan"), "mae": float("nan")})
            continue
        results["quartiles"].append({
            "quartile": i + 1,
            "n_samples": int(mask.sum()),
            "avg_predictability": float(pred_scores[mask].mean()),
            "mse": mse(preds[mask], targets[mask]),
            "mae": mae(preds[mask], targets[mask]),
        })
    return results


def error_decomposition(preds: np.ndarray, targets: np.ndarray, pred_scores: np.ndarray) -> dict:
    """Decompose total MSE into aleatoric, epistemic, and structural components.

    Aleatoric = error in Q1 (irreducible noise floor).
    Structural = excess error in Q4 beyond estimated noise floor.
    Epistemic = remainder.
    """
    strat = stratified_metrics(preds, targets, pred_scores)
    q = strat["quartiles"]
    total = strat["overall_mse"]
    q1 = q[0]["mse"] if not np.isnan(q[0]["mse"]) else total
    q4 = q[3]["mse"] if not np.isnan(q[3]["mse"]) else total

    aleatoric = q1 * 0.25
    structural = max(0, q4 - q4 * 0.1) * 0.25
    epistemic = max(0, total - aleatoric - structural)

    return {
        "total_mse": total,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "structural": structural,
        "aleatoric_frac": aleatoric / (total + 1e-10),
        "epistemic_frac": epistemic / (total + 1e-10),
        "structural_frac": structural / (total + 1e-10),
    }
