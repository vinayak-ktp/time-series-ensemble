import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Aliases for backwards compatibility with old pickle files
import src.models.arima as arima
import src.models.lgbm as lgbm
import src.models.xgboost as xgboost

sys.modules["src.models.arima_model"] = arima
sys.modules["src.models.lgbm_model"] = lgbm
sys.modules["src.models.xgboost_model"] = xgboost

MODELS_DIR = os.getenv("MODELS_DIR", "models")

LAG_PERIODS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [6, 12, 24, 168]
MIN_HISTORY = max(LAG_PERIODS)  # 168 — one week


class ForecastPredictor:
    def __init__(self):
        self.ridge = None
        self.lgbm = None
        self.xgboost = None
        self.catboost = None
        self.extra_trees = None
        self._loaded = False

    def load(self):
        try:
            with open(os.path.join(MODELS_DIR, "ridge.pkl"), "rb") as f:
                self.ridge = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "lgbm.pkl"), "rb") as f:
                self.lgbm = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
                self.xgboost = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "catboost.pkl"), "rb") as f:
                self.catboost = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "extra_trees.pkl"), "rb") as f:
                self.extra_trees = pickle.load(f)
            self._loaded = True
            print("[predictor] All models loaded successfully.")
        except FileNotFoundError as e:
            print(f"[predictor] Warning: {e}. Run training pipeline first.")
            self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def _make_time_features(self, df, datetime_col):
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

    def _build_feature_matrix_from_history(
        self,
        future_dates,
        history,
    ):
        steps = len(future_dates)
        ot_series = list(history)

        rows = []
        for i in range(steps):
            row = {}
            current_len = len(ot_series)

            for lag in LAG_PERIODS:
                if lag <= current_len:
                    row[f"OT_lag_{lag}"] = ot_series[current_len - lag]
                else:
                    row[f"OT_lag_{lag}"] = ot_series[0] if ot_series else 0.0

            for w in ROLLING_WINDOWS:
                window_vals = np.array(ot_series[max(0, current_len - w) : current_len])
                row[f"OT_roll_mean_{w}"] = (
                    float(window_vals.mean()) if len(window_vals) else 0.0
                )
                row[f"OT_roll_std_{w}"] = (
                    float(window_vals.std()) if len(window_vals) > 1 else 0.0
                )
                row[f"OT_roll_min_{w}"] = (
                    float(window_vals.min()) if len(window_vals) else 0.0
                )
                row[f"OT_roll_max_{w}"] = (
                    float(window_vals.max()) if len(window_vals) else 0.0
                )

            for col in ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]:
                row[col] = 0.0

            row["date"] = future_dates[i]
            row["OT"] = 0.0  # placeholder
            rows.append(row)
            ot_series.append(0.0)

        feat_df = pd.DataFrame(rows)
        feat_df = self._make_time_features(feat_df, "date")
        return feat_df

    def _build_feature_matrix_synthetic(self, future_dates, seed_value):
        n = len(future_dates)
        seed_series = np.full(n, seed_value)

        feat_df = pd.DataFrame({"date": future_dates, "OT": seed_series})
        feat_df = self._make_time_features(feat_df, "date")
        for lag in LAG_PERIODS:
            feat_df[f"OT_lag_{lag}"] = seed_value
        for w in ROLLING_WINDOWS:
            feat_df[f"OT_roll_mean_{w}"] = seed_value
            feat_df[f"OT_roll_std_{w}"] = 0.0
            feat_df[f"OT_roll_min_{w}"] = seed_value
            feat_df[f"OT_roll_max_{w}"] = seed_value
        for col in ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]:
            feat_df[col] = 0.0
        return feat_df

    def _model_map(self):
        return {
            "ridge": self.ridge,
            "lgbm": self.lgbm,
            "xgboost": self.xgboost,
            "catboost": self.catboost,
            "extra_trees": self.extra_trees,
        }

    def predict(
        self,
        start_datetime,
        steps=24,
        include_components=False,
        history=None,
    ):
        if not self._loaded:
            raise RuntimeError("Models not loaded. Run the training pipeline first.")

        start_dt = datetime.fromisoformat(start_datetime)
        future_dates = [start_dt + timedelta(hours=i) for i in range(steps)]

        # Hybrid Architecture: Ridge is base, CatBoost and ExtraTrees are residuals
        base_model = "ridge"
        residual_models = ["catboost", "extra_trees"]
        model_map = self._model_map()

        use_real_history = history is not None and len(history) > 0
        history_mode = "real" if use_real_history else "synthetic"

        preds_dict = {}
        hybrid_preds = []

        if use_real_history:
            ot_series = list(history)
            step_preds = {name: [] for name in [base_model] + residual_models}

            for i in range(steps):
                feat_df = self._build_feature_matrix_from_history(
                    [future_dates[i]], ot_series
                )

                # Base Trend Prediction
                try:
                    base_val = model_map[base_model].predict(feat_df)[0]
                except Exception:
                    base_val = (
                        float(np.mean(list(ot_series[-6:]))) if ot_series else 0.0
                    )
                step_preds[base_model].append(base_val)

                # Residual Predictions
                res_vals = []
                for name in residual_models:
                    try:
                        val = model_map[name].predict(feat_df)[0]
                        step_preds[name].append(val)
                        res_vals.append(val)
                    except Exception:
                        step_preds[name].append(0.0)
                        res_vals.append(0.0)

                # Hybrid Combo: Base + Mean(Residuals)
                next_ot = base_val + float(np.mean(res_vals)) if res_vals else base_val
                hybrid_preds.append(next_ot)
                ot_series.append(next_ot)

            preds_dict = {name: np.array(vals) for name, vals in step_preds.items()}

        else:
            feat_df = self._build_feature_matrix_synthetic(future_dates, seed_value=0.0)

            # Base
            try:
                preds_dict[base_model] = model_map[base_model].predict(feat_df)
            except Exception:
                preds_dict[base_model] = np.zeros(steps)

            # Residuals
            for name in residual_models:
                try:
                    preds_dict[name] = model_map[name].predict(feat_df)
                except Exception:
                    preds_dict[name] = np.zeros(steps)

            hybrid_res = np.mean([preds_dict[n] for n in residual_models], axis=0)
            hybrid_preds = preds_dict[base_model] + hybrid_res

        result = {
            "forecast": [],
            "hybrid_components": {"base": base_model, "residuals": residual_models},
            "history_mode": history_mode,
        }
        if use_real_history:
            result["history_length"] = len(history)

        for i, dt_val in enumerate(future_dates):
            point = {
                "datetime": dt_val.isoformat(),
                "prediction": float(hybrid_preds[i]),
            }
            if include_components:
                point[base_model] = float(preds_dict[base_model][i])
                for name in residual_models:
                    point[name] = float(preds_dict[name][i])
            result["forecast"].append(point)

        return result


predictor = ForecastPredictor()
