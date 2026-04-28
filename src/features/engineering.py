"""
Feature Engineering Script
Creates lag features, rolling statistics, and calendar features
for gradient boosting models (LightGBM, XGBoost).
"""
import argparse
import os
import yaml
import pandas as pd
import numpy as np


def make_lag_features(
    df: pd.DataFrame, target_col: str, lag_periods: list[int]
) -> pd.DataFrame:
    for lag in lag_periods:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def make_rolling_features(
    df: pd.DataFrame, target_col: str, windows: list[int]
) -> pd.DataFrame:
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = (
            df[target_col].shift(1).rolling(w).mean()
        )
        df[f"{target_col}_roll_std_{w}"] = (
            df[target_col].shift(1).rolling(w).std()
        )
        df[f"{target_col}_roll_min_{w}"] = (
            df[target_col].shift(1).rolling(w).min()
        )
        df[f"{target_col}_roll_max_{w}"] = (
            df[target_col].shift(1).rolling(w).max()
        )
    return df


def make_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[datetime_col])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    df["week"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def featurize(
    df: pd.DataFrame,
    target_col: str,
    datetime_col: str,
    lag_periods: list[int],
    rolling_windows: list[int],
    use_time_features: bool,
) -> pd.DataFrame:
    df = df.copy()
    df = make_lag_features(df, target_col, lag_periods)
    df = make_rolling_features(df, target_col, rolling_windows)
    if use_time_features:
        df = make_time_features(df, datetime_col)
    # Drop rows with NaN from lagging
    max_lag = max(lag_periods) if lag_periods else 0
    max_win = max(rolling_windows) if rolling_windows else 0
    drop_rows = max(max_lag, max_win)
    df = df.iloc[drop_rows:].reset_index(drop=True)
    return df


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    target_col = cfg["base"]["target_col"]
    datetime_col = cfg["base"]["datetime_col"]
    lag_periods = cfg["features"]["lag_periods"]
    rolling_windows = cfg["features"]["rolling_windows"]
    use_time_features = cfg["features"]["use_time_features"]

    splits = {
        "train": (
            cfg["data"]["processed_train_path"],
            "data/processed/train_features.csv",
        ),
        "val": (
            cfg["data"]["processed_val_path"],
            "data/processed/val_features.csv",
        ),
        "test": (
            cfg["data"]["processed_test_path"],
            "data/processed/test_features.csv",
        ),
    }

    os.makedirs("data/processed", exist_ok=True)

    for split_name, (in_path, out_path) in splits.items():
        df = pd.read_csv(in_path)
        df_feat = featurize(
            df, target_col, datetime_col, lag_periods, rolling_windows, use_time_features
        )
        df_feat.to_csv(out_path, index=False)
        print(
            f"[featurize] {split_name}: {len(df_feat)} rows, {len(df_feat.columns)} features → {out_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
