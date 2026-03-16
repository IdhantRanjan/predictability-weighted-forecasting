"""TMLR-formatted plotting utilities."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


# TMLR-compatible style settings
TMLR_STYLE = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (6.5, 4.0),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Color palette (colorblind-friendly)
COLORS = {
    "s4": "#1f77b4",
    "mamba": "#ff7f0e",
    "dlinear": "#2ca02c",
    "nlinear": "#7f7f7f",
    "patchtst": "#d62728",
    "s4_pwt": "#17becf",
    "mamba_pwt": "#bcbd22",
}

MODEL_LABELS = {
    "s4": "S4",
    "mamba": "Mamba",
    "dlinear": "DLinear",
    "nlinear": "NLinear",
    "patchtst": "PatchTST",
    "s4_pwt": "S4 + PWT",
    "mamba_pwt": "Mamba + PWT",
}


def set_tmlr_style():
    """Apply TMLR-compatible matplotlib style."""
    plt.rcParams.update(TMLR_STYLE)


def plot_stratified_performance(results: dict, save_path: str, dataset_name: str = ""):
    """Bar chart of MSE across predictability quartiles for each model.

    Args:
        results: {model_name: stratified_metrics_dict}
        save_path: path to save figure
        dataset_name: for title
    """
    set_tmlr_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    models = list(results.keys())
    n_models = len(models)
    n_quartiles = len(results[models[0]]["quartiles"])

    x = np.arange(n_quartiles)
    width = 0.8 / n_models

    for i, model in enumerate(models):
        mses = [q["mse"] for q in results[model]["quartiles"]]
        color = COLORS.get(model, f"C{i}")
        label = MODEL_LABELS.get(model, model)
        ax.bar(x + i * width - 0.4 + width / 2, mses, width, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Predictability Quartile")
    ax.set_ylabel("MSE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}\n(least)" if i == 0 else f"Q{i+1}\n(most)" if i == n_quartiles - 1 else f"Q{i+1}" for i in range(n_quartiles)])
    ax.legend(loc="upper right", framealpha=0.9)
    if dataset_name:
        ax.set_title(f"Stratified Performance — {dataset_name}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_error_decomposition(decompositions: dict, save_path: str):
    """Stacked bar chart of error decomposition across models.

    Args:
        decompositions: {model_name: error_decomposition_dict}
        save_path: path to save figure
    """
    set_tmlr_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    models = list(decompositions.keys())
    labels = [MODEL_LABELS.get(m, m) for m in models]
    x = np.arange(len(models))

    aleatoric = [decompositions[m]["aleatoric"] for m in models]
    epistemic = [decompositions[m]["epistemic"] for m in models]
    structural = [decompositions[m]["structural"] for m in models]

    ax.bar(x, aleatoric, label="Aleatoric", color="#8da0cb", alpha=0.85)
    ax.bar(x, epistemic, bottom=aleatoric, label="Epistemic", color="#fc8d62", alpha=0.85)
    bottoms = [a + e for a, e in zip(aleatoric, epistemic)]
    ax.bar(x, structural, bottom=bottoms, label="Structural", color="#66c2a5", alpha=0.85)

    ax.set_ylabel("MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Forecast Error Decomposition")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_horizon_comparison(results: dict, save_path: str, dataset_name: str = ""):
    """Line plot of MSE vs prediction horizon for each model.

    Args:
        results: {model_name: {horizon: mse_value}}
        save_path: path to save figure
        dataset_name: for title
    """
    set_tmlr_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for model, horizon_results in results.items():
        horizons = sorted(horizon_results.keys())
        mses = [horizon_results[h] for h in horizons]
        color = COLORS.get(model, None)
        label = MODEL_LABELS.get(model, model)
        ax.plot(horizons, mses, "o-", label=label, color=color, markersize=5)

    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("MSE")
    ax.legend(loc="upper left", framealpha=0.9)
    if dataset_name:
        ax.set_title(f"MSE vs Horizon — {dataset_name}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pwt_ablation(results: dict, save_path: str):
    """Grouped bar chart: baseline vs PWT for each model/dataset.

    Args:
        results: {dataset: {model: {"baseline": mse, "pwt": mse}}}
    """
    set_tmlr_style()
    datasets = list(results.keys())
    models = list(results[datasets[0]].keys())

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5, 3.0), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        x = np.arange(len(models))
        baselines = [results[ds][m]["baseline"] for m in models]
        pwts = [results[ds][m]["pwt"] for m in models]

        ax.bar(x - 0.2, baselines, 0.35, label="Baseline", color="#8da0cb", alpha=0.85)
        ax.bar(x + 0.2, pwts, 0.35, label="+ PWT", color="#fc8d62", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=30, ha="right")
        ax.set_title(ds)
        if ax == axes[0]:
            ax.set_ylabel("MSE")
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_predictability_timeseries(scores: np.ndarray, save_path: str, dataset_name: str = ""):
    """Plot predictability scores over time."""
    set_tmlr_style()
    fig, ax = plt.subplots(figsize=(6.5, 2.5))

    ax.plot(scores, linewidth=0.5, color="#1f77b4", alpha=0.7)
    ax.fill_between(range(len(scores)), scores, alpha=0.2, color="#1f77b4")

    # Mark quartile boundaries
    for q in [25, 50, 75]:
        val = np.percentile(scores, q)
        ax.axhline(val, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.text(len(scores) * 0.98, val, f"Q{q//25}", fontsize=7, ha="right", va="bottom", color="gray")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Predictability Score")
    ax.set_ylim(0, 1)
    if dataset_name:
        ax.set_title(f"Local Predictability — {dataset_name}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
