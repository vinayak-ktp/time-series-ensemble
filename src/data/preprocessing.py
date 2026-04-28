"""
Preprocessing Script
Splits raw time series data into train/val/test splits in chronological order.
Applies scaling and saves processed CSVs.
"""
import argparse
import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


def load_raw(path: str, datetime_col: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)
    # Keep only datetime + target for univariate forecasting
    # (keep all columns for multi-variate feature engineering)
    return df


def chronological_split(
    df: pd.DataFrame, test_size: float, val_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()

    print(
        f"[preprocess] Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}"
    )
    return train, val, test


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_path = cfg["data"]["raw_path"]
    target_col = cfg["base"]["target_col"]
    datetime_col = cfg["base"]["datetime_col"]
    test_size = cfg["base"]["test_size"]
    val_size = cfg["base"]["val_size"]
    train_out = cfg["data"]["processed_train_path"]
    val_out = cfg["data"]["processed_val_path"]
    test_out = cfg["data"]["processed_test_path"]

    os.makedirs("data/processed", exist_ok=True)

    df = load_raw(raw_path, datetime_col, target_col)
    train, val, test = chronological_split(df, test_size, val_size)

    # Fit scaler on train, transform all splits
    feature_cols = [c for c in df.columns if c != datetime_col]
    scaler = StandardScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    val[feature_cols] = scaler.transform(val[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    # Save scaler for inference
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    train.to_csv(train_out, index=False)
    val.to_csv(val_out, index=False)
    test.to_csv(test_out, index=False)

    print(f"[preprocess] Saved train → {train_out}")
    print(f"[preprocess] Saved val   → {val_out}")
    print(f"[preprocess] Saved test  → {test_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
