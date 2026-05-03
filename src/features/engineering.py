import argparse
import os

import numpy as np
import pandas as pd
import yaml


def make_lag_features(df, target_col, lag_periods):
    for lag in lag_periods:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def make_rolling_features(df, target_col, windows):
    for w in windows:
        shifted = df[target_col].shift(1)
        df[f"{target_col}_roll_mean_{w}"] = shifted.rolling(w).mean()
        df[f"{target_col}_roll_std_{w}"] = shifted.rolling(w).std()
        df[f"{target_col}_roll_min_{w}"] = shifted.rolling(w).min()
        df[f"{target_col}_roll_max_{w}"] = shifted.rolling(w).max()
    return df


def make_ewm_features(df, target_col, spans):
    """Exponentially weighted mean — recency-weighted trend signal for tree residual models."""
    for span in spans:
        df[f"{target_col}_ewm_{span}"] = (
            df[target_col].shift(1).ewm(span=span, adjust=False).mean()
        )
    return df


def make_diff_features(df, target_col, lag_pairs):
    """Velocity/direction features: lag_a - lag_b. Models rate of change explicitly."""
    for a, b in lag_pairs:
        col_a = f"{target_col}_lag_{a}"
        col_b = f"{target_col}_lag_{b}"
        if col_a in df.columns and col_b in df.columns:
            df[f"{target_col}_diff_{a}_{b}"] = df[col_a] - df[col_b]
    return df


def make_time_features(df, datetime_col):
    dt = pd.to_datetime(df[datetime_col])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    df["week"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def make_interaction_features(df, target_col):
    """
    Interaction features that help tree models learn time-of-day × level effects.
    These are particularly effective for residual correction in the hybrid architecture.
    """
    if "hour_sin" in df.columns and f"{target_col}_roll_mean_24" in df.columns:
        df[f"{target_col}_hour_sin_x_mean24"] = (
            df["hour_sin"] * df[f"{target_col}_roll_mean_24"]
        )
        df[f"{target_col}_hour_cos_x_mean24"] = (
            df["hour_cos"] * df[f"{target_col}_roll_mean_24"]
        )
    if "is_weekend" in df.columns and f"{target_col}_lag_24" in df.columns:
        df[f"{target_col}_weekend_x_lag24"] = (
            df["is_weekend"] * df[f"{target_col}_lag_24"]
        )
    if "dow_sin" in df.columns and f"{target_col}_roll_mean_24" in df.columns:
        df[f"{target_col}_dow_sin_x_mean24"] = (
            df["dow_sin"] * df[f"{target_col}_roll_mean_24"]
        )
    return df


def featurize(
    df,
    target_col,
    datetime_col,
    lag_periods,
    rolling_windows,
    use_time_features,
    ewm_spans=None,
    diff_pairs=None,
    use_interactions=False,
):
    df = df.copy()
    df = make_lag_features(df, target_col, lag_periods)
    df = make_rolling_features(df, target_col, rolling_windows)
    if ewm_spans:
        df = make_ewm_features(df, target_col, ewm_spans)
    if use_time_features:
        df = make_time_features(df, datetime_col)
    if diff_pairs:
        df = make_diff_features(df, target_col, diff_pairs)
    if use_interactions and use_time_features:
        df = make_interaction_features(df, target_col)
    max_lag = max(lag_periods) if lag_periods else 0
    max_win = max(rolling_windows) if rolling_windows else 0
    max_ewm = 0  # EWM doesn't create NaNs beyond first row
    drop_rows = max(max_lag, max_win, max_ewm)
    df = df.iloc[drop_rows:].reset_index(drop=True)
    return df


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    target_col = cfg["base"]["target_col"]
    datetime_col = cfg["base"]["datetime_col"]
    feat_cfg = cfg["features"]
    lag_periods = feat_cfg["lag_periods"]
    rolling_windows = feat_cfg["rolling_windows"]
    use_time_features = feat_cfg["use_time_features"]
    ewm_spans = feat_cfg.get("ewm_spans", None)
    diff_pairs = [tuple(p) for p in feat_cfg.get("diff_pairs", [])]
    use_interactions = feat_cfg.get("use_interactions", False)

    splits = {
        "train": (
            cfg["data"]["processed_train_path"],
            "data/processed/train_features.csv",
        ),
        "val": (cfg["data"]["processed_val_path"], "data/processed/val_features.csv"),
        "test": (
            cfg["data"]["processed_test_path"],
            "data/processed/test_features.csv",
        ),
    }

    os.makedirs("data/processed", exist_ok=True)

    for split_name, (in_path, out_path) in splits.items():
        df = pd.read_csv(in_path)
        df_feat = featurize(
            df,
            target_col,
            datetime_col,
            lag_periods,
            rolling_windows,
            use_time_features,
            ewm_spans,
            diff_pairs,
            use_interactions,
        )
        df_feat.to_csv(out_path, index=False)
        print(
            f"[featurize] {split_name}: {len(df_feat)} rows, "
            f"{len(df_feat.columns)} features → {out_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
