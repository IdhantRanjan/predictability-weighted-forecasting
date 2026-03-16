# Predictability-Stratified Evaluation and Training of State Space Models

Code for the paper:

> **Decomposing the Forecastable: Predictability-Stratified Evaluation and Training of State Space Models for Time Series**
> Under review at TMLR.

## Overview

Standard time-series benchmarks report average MSE across all test windows, mixing inherently unpredictable segments with structured ones. This repo provides:

- A per-window **predictability score** based on spectral entropy, used to stratify evaluation into predictability quartiles
- A **forecast error decomposition** into aleatoric, epistemic, and structural components
- **Predictability-Weighted Training (PWT)**, a curriculum strategy that upweights loss on high-predictability windows

We evaluate DLinear, NLinear, PatchTST, S4, and Mamba on ETT benchmarks. PWT improves SSM performance by up to 18% MSE at longer horizons.

## Setup

```bash
pip install -r requirements.txt
```

Data (ETTh1/h2, ETTm1/m2) should be placed in `data/raw/`. Download from [ETDataset](https://github.com/zhouhaoyi/ETDataset).

## Reproducing results

```bash
# Compute predictability scores and run all experiments
python run_experiments.py --dataset ETTh1 --model s4 --pred_len 96
python run_experiments.py --dataset ETTh1 --model s4 --pred_len 96 --pwt

# Generate figures and LaTeX tables from saved results
python generate_all.py

# Compile paper (requires tectonic or pdflatex)
cd paper && tectonic main.tex
```

## Structure

```
src/
  models/       # S4, Mamba, PatchTST, DLinear, NLinear
  data/         # Dataset loader, predictability scoring
  utils/        # Metrics, error decomposition, plotting
paper/          # LaTeX source (TMLR format)
figures/        # Generated figures
```

## Citation

```bibtex
@article{ranjan2025predictability,
  title={Decomposing the Forecastable: Predictability-Stratified Evaluation and Training of State Space Models for Time Series},
  author={Ranjan, Idhant},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```
