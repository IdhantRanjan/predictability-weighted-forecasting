#!/usr/bin/env python3
"""Generate all figures, tables, and compile paper from existing results."""
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, 'results')
FIGURES_DIR = os.path.join(BASE, 'figures')
TABLES_DIR = os.path.join(BASE, 'paper', 'tables')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Load all results
results = []
for f in glob.glob(os.path.join(RESULTS_DIR, '*/*/*/results.json')):
    with open(f) as fh:
        d = json.load(fh)
    parts = f.replace(RESULTS_DIR + '/', '').split('/')
    d['model_dir'] = parts[0]
    results.append(d)
print(f"Loaded {len(results)} results")

# Organize by (model_dir, dataset, pred_len)
by_key = {}
for r in results:
    key = (r['model_dir'], r['dataset'], r['pred_len'])
    by_key[key] = r

MODELS = ['dlinear', 'nlinear', 'patchtst', 's4', 'mamba']
PWT_MODELS = ['s4_pwt', 'mamba_pwt', 'patchtst_pwt']
ALL_MODELS = MODELS + PWT_MODELS
DATASETS = ['ETTh1', 'ETTh2']
HORIZONS = [96, 192, 336, 720]
DISPLAY = {
    'dlinear': 'DLinear', 'nlinear': 'NLinear', 'patchtst': 'PatchTST',
    's4': 'S4', 'mamba': 'Mamba', 's4_pwt': 'S4+PWT', 'mamba_pwt': 'Mamba+PWT',
    'patchtst_pwt': 'PatchTST+PWT'
}
COLORS = {
    'dlinear': '#94a3b8', 'nlinear': '#64748b', 'patchtst': '#f59e0b',
    's4': '#3b82f6', 'mamba': '#10b981', 's4_pwt': '#1d4ed8',
    'mamba_pwt': '#047857', 'patchtst_pwt': '#d97706'
}

def get_mse(model, dataset, horizon):
    r = by_key.get((model, dataset, horizon))
    if r is None: return None
    return r.get('test_mse', r.get('mse'))

def get_stratified(model, dataset, horizon):
    r = by_key.get((model, dataset, horizon))
    if r is None or 'stratified' not in r: return None
    return r['stratified']

def get_decomp(model, dataset, horizon):
    r = by_key.get((model, dataset, horizon))
    if r is None or 'decomposition' not in r: return None
    return r['decomposition']

# =========================================
# FIGURE 1: Stratified bar chart (ETTh1 h96)
# =========================================
print("Generating Figure 1: Stratified analysis...")
fig, ax = plt.subplots(figsize=(10, 5))
models_to_plot = ['dlinear', 'nlinear', 'patchtst', 's4', 'mamba', 's4_pwt', 'mamba_pwt']
x = np.arange(4)  # 4 quartiles
width = 0.11
for i, m in enumerate(models_to_plot):
    strat = get_stratified(m, 'ETTh1', 96)
    if strat is None: continue
    mses = [float(q['mse']) for q in strat['quartiles']]
    ax.bar(x + i*width, mses, width, label=DISPLAY[m], color=COLORS[m], edgecolor='white', linewidth=0.5)
ax.set_xlabel('Predictability Quartile')
ax.set_ylabel('MSE')
ax.set_title('Stratified Performance by Predictability (ETTh1, H=96)')
ax.set_xticks(x + width * len(models_to_plot) / 2)
ax.set_xticklabels(['Q1\n(Least)', 'Q2', 'Q3', 'Q4\n(Most)'])
ax.legend(ncol=4, loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, 'stratified_etth1.pdf'))
plt.savefig(os.path.join(FIGURES_DIR, 'stratified_etth1.png'))
plt.close()
print("  -> stratified_etth1.pdf")

# =========================================
# FIGURE 2: Error decomposition
# =========================================
print("Generating Figure 2: Error decomposition...")
fig, ax = plt.subplots(figsize=(10, 5))
models_decomp = ['dlinear', 'nlinear', 'patchtst', 's4', 'mamba', 's4_pwt', 'mamba_pwt']
x = np.arange(len(models_decomp))
aleatoric, epistemic, structural = [], [], []
for m in models_decomp:
    d = get_decomp(m, 'ETTh1', 96)
    if d:
        aleatoric.append(float(d['aleatoric']))
        epistemic.append(float(d['epistemic']))
        structural.append(float(d['structural']))
    else:
        aleatoric.append(0); epistemic.append(0); structural.append(0)

ax.bar(x, aleatoric, label='Aleatoric (irreducible)', color='#94a3b8')
ax.bar(x, epistemic, bottom=aleatoric, label='Epistemic (capacity)', color='#f59e0b')
ax.bar(x, structural, bottom=[a+e for a,e in zip(aleatoric, epistemic)], label='Structural (dynamics)', color='#ef4444')
ax.set_xticks(x)
ax.set_xticklabels([DISPLAY[m] for m in models_decomp], rotation=30, ha='right')
ax.set_ylabel('MSE Contribution')
ax.set_title('Forecast Error Decomposition (ETTh1, H=96)')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, 'decomposition.pdf'))
plt.savefig(os.path.join(FIGURES_DIR, 'decomposition.png'))
plt.close()
print("  -> decomposition.pdf")

# =========================================
# FIGURE 3: PWT ablation (base vs PWT for SSMs)
# =========================================
print("Generating Figure 3: PWT ablation...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, dataset in enumerate(DATASETS):
    ax = axes[idx]
    pairs = [('s4', 's4_pwt'), ('mamba', 'mamba_pwt'), ('patchtst', 'patchtst_pwt')]
    x = np.arange(4)
    width = 0.12
    for i, (base, pwt) in enumerate(pairs):
        base_mses = [get_mse(base, dataset, h) or 0 for h in HORIZONS]
        pwt_mses = [get_mse(pwt, dataset, h) or 0 for h in HORIZONS]
        ax.bar(x + i*2*width, base_mses, width, label=f'{DISPLAY[base]}', color=COLORS[base], alpha=0.6)
        ax.bar(x + (i*2+1)*width, pwt_mses, width, label=f'{DISPLAY[pwt]}', color=COLORS[pwt])
    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel('MSE')
    ax.set_title(f'PWT Effect on {dataset}')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'pwt_ablation.pdf'))
plt.savefig(os.path.join(FIGURES_DIR, 'pwt_ablation.png'))
plt.close()
print("  -> pwt_ablation.pdf")

# =========================================
# FIGURE 4: Horizon analysis
# =========================================
print("Generating Figure 4: Horizon analysis...")
fig, ax = plt.subplots(figsize=(8, 5))
for m in ['dlinear', 'patchtst', 's4', 'mamba', 's4_pwt', 'mamba_pwt']:
    mses = [get_mse(m, 'ETTh1', h) for h in HORIZONS]
    if None in mses: continue
    ls = '--' if '_pwt' in m else '-'
    marker = 'o' if '_pwt' not in m else 's'
    ax.plot(HORIZONS, mses, ls, marker=marker, label=DISPLAY[m], color=COLORS[m], linewidth=2, markersize=6)
ax.set_xlabel('Prediction Horizon')
ax.set_ylabel('MSE')
ax.set_title('MSE vs Prediction Horizon (ETTh1)')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, 'horizon_etth1.pdf'))
plt.savefig(os.path.join(FIGURES_DIR, 'horizon_etth1.png'))
plt.close()
print("  -> horizon_etth1.pdf")

# =========================================
# FIGURE 5: Stratified all datasets
# =========================================
print("Generating Figure 5: Stratified all datasets...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, dataset in enumerate(DATASETS):
    ax = axes[idx]
    models_to_plot = ['dlinear', 'patchtst', 's4', 'mamba', 's4_pwt', 'mamba_pwt']
    x = np.arange(4)
    width = 0.13
    for i, m in enumerate(models_to_plot):
        strat = get_stratified(m, dataset, 96)
        if strat is None: continue
        mses = [float(q['mse']) for q in strat['quartiles']]
        ax.bar(x + i*width, mses, width, label=DISPLAY[m], color=COLORS[m], edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Predictability Quartile')
    ax.set_ylabel('MSE')
    ax.set_title(f'{dataset} (H=96)')
    ax.set_xticks(x + width * len(models_to_plot) / 2)
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax.legend(fontsize=7, ncol=3)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Stratified Performance Across Datasets', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'stratified_all_datasets.pdf'))
plt.savefig(os.path.join(FIGURES_DIR, 'stratified_all_datasets.png'))
plt.close()
print("  -> stratified_all_datasets.pdf")

# =========================================
# FIGURE 6: Predictability scores over time
# =========================================
print("Generating Figure 6: Predictability visualization...")
import pandas as pd
from scipy.signal import welch

data_path = os.path.join(BASE, 'data', 'raw', 'ETTh1.csv')
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    col = df.columns[1]  # first feature
    vals = df[col].values.astype(float)
    W, S = 48, 24
    scores = []
    positions = []
    for start in range(0, len(vals) - W, S):
        seg = vals[start:start+W]
        if np.std(seg) < 1e-8:
            scores.append(0.5)
        else:
            f, psd = welch(seg, nperseg=min(32, len(seg)))
            psd = psd + 1e-12
            p = psd / psd.sum()
            se = -np.sum(p * np.log(p)) / np.log(len(p))
            scores.append(1.0 - se)
        positions.append(start + W // 2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1.2]})
    ax1.plot(vals[:2000], color='#334155', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel(col)
    ax1.set_title('ETTh1 Time Series and Local Predictability Score')
    ax1.set_xlim(0, 2000)

    pos_arr = np.array(positions)
    sc_arr = np.array(scores)
    mask = pos_arr < 2000
    ax2.fill_between(pos_arr[mask], sc_arr[mask], alpha=0.3, color='#3b82f6')
    ax2.plot(pos_arr[mask], sc_arr[mask], color='#1d4ed8', linewidth=1)
    q25, q50, q75 = np.percentile(sc_arr, [25, 50, 75])
    for q, lbl in [(q25, 'Q1/Q2'), (q50, 'Q2/Q3'), (q75, 'Q3/Q4')]:
        ax2.axhline(q, color='#ef4444', linestyle='--', alpha=0.5, linewidth=0.8)
        ax2.text(1950, q+0.01, lbl, fontsize=7, color='#ef4444')
    ax2.set_ylabel('Predictability Score π')
    ax2.set_xlabel('Timestep')
    ax2.set_xlim(0, 2000)
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'predictability_etth1.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'predictability_etth1.png'))
    plt.close()
    print("  -> predictability_etth1.pdf")
else:
    print("  -> SKIPPED (no data file)")

# =========================================
# TABLE: Main results LaTeX
# =========================================
print("Generating LaTeX tables...")

# Build main results table
lines = []
lines.append(r'\begin{tabular}{ll' + 'c' * len(HORIZONS) + '}')
lines.append(r'\toprule')
lines.append(r'\textbf{Dataset} & \textbf{Model} & ' + ' & '.join([f'$H={h}$' for h in HORIZONS]) + r' \\')
lines.append(r'\midrule')

for dataset in DATASETS:
    all_models = ['dlinear', 'nlinear', 'patchtst', 's4', 's4_pwt', 'mamba', 'mamba_pwt', 'patchtst_pwt']
    # Find best MSE per horizon
    best = {}
    for h in HORIZONS:
        vals = [(m, get_mse(m, dataset, h)) for m in all_models if get_mse(m, dataset, h) is not None]
        if vals:
            best[h] = min(vals, key=lambda x: x[1])[1]

    for mi, m in enumerate(all_models):
        prefix = f'\\multirow{{{len(all_models)}}}{{*}}{{{dataset}}}' if mi == 0 else ''
        cells = []
        for h in HORIZONS:
            v = get_mse(m, dataset, h)
            if v is None:
                cells.append('--')
            elif abs(v - best.get(h, -1)) < 0.0001:
                cells.append(f'\\textbf{{{v:.4f}}}')
            else:
                cells.append(f'{v:.4f}')
        lines.append(f'{prefix} & {DISPLAY[m]} & ' + ' & '.join(cells) + r' \\')
    lines.append(r'\midrule')

lines[-1] = r'\bottomrule'
lines.append(r'\end{tabular}')

with open(os.path.join(TABLES_DIR, 'main_results.tex'), 'w') as f:
    f.write('\n'.join(lines))
print("  -> main_results.tex")

# =========================================
# TABLE: PWT improvement summary
# =========================================
lines2 = []
lines2.append(r'\begin{tabular}{llcccc}')
lines2.append(r'\toprule')
lines2.append(r'\textbf{Model} & \textbf{Dataset} & \textbf{Avg MSE Base} & \textbf{Avg MSE PWT} & \textbf{$\Delta$ MSE} & \textbf{$\Delta$ \%} \\')
lines2.append(r'\midrule')

for base, pwt in [('s4', 's4_pwt'), ('mamba', 'mamba_pwt'), ('patchtst', 'patchtst_pwt')]:
    for dataset in DATASETS:
        base_vals = [get_mse(base, dataset, h) for h in HORIZONS]
        pwt_vals = [get_mse(pwt, dataset, h) for h in HORIZONS]
        if None in base_vals or None in pwt_vals: continue
        avg_base = np.mean(base_vals)
        avg_pwt = np.mean(pwt_vals)
        delta = avg_pwt - avg_base
        pct = (delta / avg_base) * 100
        lines2.append(f'{DISPLAY[base]} & {dataset} & {avg_base:.4f} & {avg_pwt:.4f} & {delta:+.4f} & {pct:+.1f}\\% \\\\')
lines2.append(r'\bottomrule')
lines2.append(r'\end{tabular}')

with open(os.path.join(TABLES_DIR, 'pwt_improvement.tex'), 'w') as f:
    f.write('\n'.join(lines2))
print("  -> pwt_improvement.tex")

# =========================================
# SUMMARY STATS for paper
# =========================================
print("\n" + "="*60)
print("SUMMARY OF KEY RESULTS")
print("="*60)
for base, pwt in [('s4', 's4_pwt'), ('mamba', 'mamba_pwt'), ('patchtst', 'patchtst_pwt')]:
    improvements = []
    for dataset in DATASETS:
        for h in HORIZONS:
            b = get_mse(base, dataset, h)
            p = get_mse(pwt, dataset, h)
            if b and p:
                improvements.append((p - b) / b * 100)
    if improvements:
        print(f"{DISPLAY[base]} -> {DISPLAY[pwt]}: avg {np.mean(improvements):+.1f}%, range [{min(improvements):+.1f}%, {max(improvements):+.1f}%]")

print("\nBest overall model per horizon (ETTh1):")
for h in HORIZONS:
    vals = [(m, get_mse(m, 'ETTh1', h)) for m in ALL_MODELS if get_mse(m, 'ETTh1', h)]
    if vals:
        best_m, best_v = min(vals, key=lambda x: x[1])
        print(f"  H={h}: {DISPLAY[best_m]} ({best_v:.4f})")

print("\nDecomposition (ETTh1, H=96):")
for m in ['dlinear', 's4', 's4_pwt', 'mamba', 'mamba_pwt']:
    d = get_decomp(m, 'ETTh1', 96)
    if d:
        print(f"  {DISPLAY[m]:12s}: A={float(d['aleatoric_frac'])*100:.1f}% E={float(d['epistemic_frac'])*100:.1f}% S={float(d['structural_frac'])*100:.1f}%")

print("\n" + "="*60)
print("ALL FIGURES AND TABLES GENERATED SUCCESSFULLY")
print("="*60)
