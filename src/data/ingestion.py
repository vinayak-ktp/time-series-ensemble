import argparse
import os
import urllib.request

import pandas as pd
import yaml

ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


def download_dataset(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"[ingest] Dataset already exists at {dest_path}. Skipping download.")
        return
    print(f"[ingest] Downloading ETTh1 dataset → {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print("[ingest] Download complete.")


def validate_dataset(path: str, target_col: str, datetime_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[datetime_col])
    assert target_col in df.columns, f"Target column '{target_col}' not found."
    assert datetime_col in df.columns, f"Datetime column '{datetime_col}' not found."
    assert df[datetime_col].is_monotonic_increasing, "Timestamps are not sorted."
    assert not df[target_col].isnull().all(), "Target column is entirely null."
    print(f"[ingest] Validated: {len(df)} rows, {df.columns.tolist()}")
    return df


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_path = cfg["data"]["raw_path"]
    target_col = cfg["base"]["target_col"]
    datetime_col = cfg["base"]["datetime_col"]

    download_dataset(ETT_URL, raw_path)
    df = validate_dataset(raw_path, target_col, datetime_col)
    print(f"[ingest] Date range: {df[datetime_col].min()} → {df[datetime_col].max()}")
    print(f"[ingest] Target stats:\n{df[target_col].describe()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
