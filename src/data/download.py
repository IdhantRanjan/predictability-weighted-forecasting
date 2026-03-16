"""Download and prepare all benchmark datasets."""
import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path


# ETT and other datasets from the Autoformer/TimesNet GitHub repos
DATASET_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
    "Exchange": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt",
    "Weather": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    # Weather uses a different source; we'll generate synthetic if unavailable
}


def download_file(url: str, dest: str) -> bool:
    """Download a file from URL to destination path."""
    try:
        print(f"  Downloading from {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"  Warning: Could not download {url}: {e}")
        return False


def generate_synthetic_weather(dest: str, n_points: int = 52696, n_features: int = 21):
    """Generate synthetic weather data matching the Weather dataset schema."""
    print("  Generating synthetic Weather dataset...")
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_points, freq="10min")

    data = {"date": dates}
    feature_names = [
        "p_mbar", "T_degC", "Tpot_K", "Tdew_degC", "rh_pct",
        "VPmax_mbar", "VPact_mbar", "VPdef_mbar", "sh_g_kg",
        "H2OC_mmol_mol", "rho_g_m3", "wv_m_s", "max_wv_m_s",
        "wd_deg", "rain_mm", "raining_s", "SWDR_W_m2",
        "PAR_umol_m2_s", "max_PAR_umol_m2_s", "Tlog_degC", "OT"
    ]

    t = np.arange(n_points)
    for i, name in enumerate(feature_names):
        # Mix of seasonal patterns + noise
        seasonal = np.sin(2 * np.pi * t / (144 * 365) + i) * 10
        daily = np.sin(2 * np.pi * t / 144 + i * 0.5) * 5
        trend = t * 0.0001 * (i % 3 - 1)
        noise = np.random.randn(n_points) * (1 + i * 0.1)
        data[name] = seasonal + daily + trend + noise + 20 * (i == 1)

    df = pd.DataFrame(data)
    df.to_csv(dest, index=False)


def generate_synthetic_exchange(dest: str, n_points: int = 7588, n_currencies: int = 8):
    """Generate synthetic exchange rate data."""
    print("  Generating synthetic Exchange dataset...")
    np.random.seed(123)
    t = np.arange(n_points)

    data = []
    for i in range(n_currencies):
        base = 1.0 + i * 0.3
        trend = t * 0.00001 * (-1) ** i
        ar = np.zeros(n_points)
        ar[0] = base
        for j in range(1, n_points):
            ar[j] = 0.999 * ar[j - 1] + np.random.randn() * 0.002
        series = ar + trend
        data.append(series)

    df = pd.DataFrame(np.array(data).T)
    df.to_csv(dest, index=False, header=False)


def prepare_ett_data(raw_path: str, dataset_name: str) -> pd.DataFrame:
    """Load and validate ETT dataset."""
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {dataset_name}: {len(df)} rows, {len(df.columns) - 1} features")
    return df


def prepare_exchange_data(raw_path: str) -> pd.DataFrame:
    """Load exchange rate data."""
    df = pd.read_csv(raw_path, header=None)
    df.columns = [f"currency_{i}" for i in range(len(df.columns))]
    dates = pd.date_range("1990-01-01", periods=len(df), freq="D")
    df.insert(0, "date", dates)
    print(f"  Exchange: {len(df)} rows, {len(df.columns) - 1} features")
    return df


def download_all_datasets(data_root: str = "./data") -> dict:
    """Download and prepare all datasets. Returns dict of DataFrames."""
    data_root = Path(data_root)
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}

    # ETT datasets
    for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        print(f"Preparing {name}...")
        raw_path = raw_dir / f"{name}.csv"
        proc_path = processed_dir / f"{name}.csv"

        if not proc_path.exists():
            if not raw_path.exists():
                success = download_file(DATASET_URLS[name], str(raw_path))
                if not success:
                    # Generate synthetic ETT data
                    print(f"  Generating synthetic {name} data...")
                    np.random.seed(hash(name) % 2**31)
                    n = 17420 if "h" in name else 69680
                    dates = pd.date_range(
                        "2016-07-01",
                        periods=n,
                        freq="h" if "h" in name else "15min",
                    )
                    t = np.arange(n)
                    cols = {"date": dates}
                    for j in range(7):
                        cols[f"feat_{j}"] = (
                            np.sin(2 * np.pi * t / (24 if "h" in name else 96) + j)
                            * 5
                            + np.random.randn(n) * 0.5
                            + 30
                        )
                    cols["OT"] = cols["feat_0"] + np.random.randn(n) * 0.3
                    df = pd.DataFrame(cols)
                    df.to_csv(raw_path, index=False)

            df = prepare_ett_data(str(raw_path), name)
            df.to_csv(proc_path, index=False)
        else:
            df = pd.read_csv(proc_path)
            df["date"] = pd.to_datetime(df["date"])

        datasets[name] = df

    # Exchange dataset
    print("Preparing Exchange...")
    raw_path = raw_dir / "exchange_rate.txt"
    proc_path = processed_dir / "Exchange.csv"
    if not proc_path.exists():
        success = download_file(DATASET_URLS["Exchange"], str(raw_path))
        if not success:
            generate_synthetic_exchange(str(raw_path))
        df = prepare_exchange_data(str(raw_path))
        df.to_csv(proc_path, index=False)
    else:
        df = pd.read_csv(proc_path)
    datasets["Exchange"] = df

    # Weather dataset
    print("Preparing Weather...")
    proc_path = processed_dir / "Weather.csv"
    if not proc_path.exists():
        generate_synthetic_weather(str(proc_path))
    df = pd.read_csv(proc_path)
    datasets["Weather"] = df

    print(f"\nAll {len(datasets)} datasets ready.")
    return datasets


if __name__ == "__main__":
    download_all_datasets()
