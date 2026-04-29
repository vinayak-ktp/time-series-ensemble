import os
import pickle
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

MODELS_DIR = os.getenv("MODELS_DIR", "models")

LAG_PERIODS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [6, 12, 24, 168]
MIN_HISTORY = max(LAG_PERIODS)  # 168 — one week


class ForecastPredictor:
    def __init__(self):
        self.arima = None
        self.ridge = None
        self.lgbm = None
        self.xgboost = None
        self.ensemble = None
        self._loaded = False

    def load(self) -> None:
        try:
            with open(os.path.join(MODELS_DIR, "arima.pkl"), "rb") as f:
                self.arima = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "ridge.pkl"), "rb") as f:
                self.ridge = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "lgbm.pkl"), "rb") as f:
                self.lgbm = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
                self.xgboost = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "ensemble.pkl"), "rb") as f:
                self.ensemble = pickle.load(f)
            self._loaded = True
            print("[predictor] All models loaded successfully.")
        except FileNotFoundError as e:
            print(f"[predictor] Warning: {e}. Run training pipeline first.")
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _make_time_features(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
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
        future_dates: List[datetime],
        history: List[float],
    ) -> pd.DataFrame:
        """
        Build a feature matrix for the forecast horizon using real historical OT values.

        Strategy: create a combined series (history + iteratively-predicted OT) to
        compute proper lag and rolling features for each forecast step. For each
        step t, all lags are taken from known values — either from history (for
        t <= MIN_HISTORY) or from prior forecast steps (the model recurses using
        its own predictions for large lags when history is shorter than needed).
        """
        steps = len(future_dates)
        ot_series = list(history)  # grows as we recurse through forecast steps

        rows = []
        for i in range(steps):
            row = {}
            current_len = len(ot_series)  # available values before step i

            # Lag features — taken from combined history+predictions so far
            for lag in LAG_PERIODS:
                if lag <= current_len:
                    row[f"OT_lag_{lag}"] = ot_series[current_len - lag]
                else:
                    # Not enough history — use the earliest available value
                    row[f"OT_lag_{lag}"] = ot_series[0] if ot_series else 0.0

            # Rolling window features
            for w in ROLLING_WINDOWS:
                window_vals = np.array(ot_series[max(0, current_len - w) : current_len])
                row[f"OT_roll_mean_{w}"] = float(window_vals.mean()) if len(window_vals) else 0.0
                row[f"OT_roll_std_{w}"] = float(window_vals.std()) if len(window_vals) > 1 else 0.0
                row[f"OT_roll_min_{w}"] = float(window_vals.min()) if len(window_vals) else 0.0
                row[f"OT_roll_max_{w}"] = float(window_vals.max()) if len(window_vals) else 0.0

            # Other sensor columns — unavailable at inference time, zero-filled
            for col in ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]:
                row[col] = 0.0

            row["date"] = future_dates[i]
            row["OT"] = 0.0  # placeholder; removed before model.predict
            rows.append(row)

            # Recurse: use the ensemble prediction as the next step's OT value.
            # We append a placeholder here and overwrite after the first model pass.
            # (See predict() for how this is handled in two-pass fashion.)
            ot_series.append(0.0)

        feat_df = pd.DataFrame(rows)
        feat_df = self._make_time_features(feat_df, "date")
        return feat_df

    def _build_feature_matrix_synthetic(
        self, future_dates: List[datetime], arima_preds: np.ndarray
    ) -> pd.DataFrame:
        """Fallback: build feature matrix using ARIMA extrapolation as synthetic OT."""
        feat_df = pd.DataFrame({"date": future_dates, "OT": arima_preds})
        feat_df = self._make_time_features(feat_df, "date")
        for lag in LAG_PERIODS:
            feat_df[f"OT_lag_{lag}"] = np.roll(arima_preds, lag)
        for w in ROLLING_WINDOWS:
            s = pd.Series(arima_preds)
            feat_df[f"OT_roll_mean_{w}"] = s.rolling(w, min_periods=1).mean().values
            feat_df[f"OT_roll_std_{w}"] = s.rolling(w, min_periods=1).std().fillna(0).values
            feat_df[f"OT_roll_min_{w}"] = s.rolling(w, min_periods=1).min().values
            feat_df[f"OT_roll_max_{w}"] = s.rolling(w, min_periods=1).max().values
        for col in ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]:
            feat_df[col] = 0.0
        return feat_df

    def predict(
        self,
        start_datetime: str,
        steps: int = 24,
        include_components: bool = False,
        history: Optional[List[float]] = None,
    ) -> dict:
        if not self._loaded:
            raise RuntimeError("Models not loaded. Run the training pipeline first.")

        start_dt = datetime.fromisoformat(start_datetime)
        future_dates = [start_dt + timedelta(hours=i) for i in range(steps)]
        future_df = pd.DataFrame({"date": future_dates})

        # ARIMA: always uses its internal state, extending from training end
        arima_preds = np.array(self.arima.predict(steps=steps))

        use_real_history = history is not None and len(history) > 0
        history_mode = "real" if use_real_history else "synthetic"

        if use_real_history:
            # Step-by-step recursive feature construction using real history
            ot_series = list(history)
            ridge_preds = np.zeros(steps)
            lgbm_preds = np.zeros(steps)
            xgb_preds = np.zeros(steps)

            for i in range(steps):
                feat_df = self._build_feature_matrix_from_history(
                    [future_dates[i]], ot_series
                )
                try:
                    ridge_preds[i] = self.ridge.predict(feat_df, "OT", "date")[0]
                    lgbm_preds[i] = self.lgbm.predict(feat_df, "OT", "date")[0]
                    xgb_preds[i] = self.xgboost.predict(feat_df, "OT", "date")[0]
                except Exception:
                    ridge_preds[i] = arima_preds[i]
                    lgbm_preds[i] = arima_preds[i]
                    xgb_preds[i] = arima_preds[i]

                # Auto-regressive rollout: use mean of tree models as next OT
                ot_series.append(float((lgbm_preds[i] + xgb_preds[i]) / 2.0))
        else:
            # Synthetic fallback: build feature matrix from ARIMA extrapolation
            arima_series = arima_preds
            feat_df = self._build_feature_matrix_synthetic(future_dates, arima_series)
            try:
                ridge_preds = self.ridge.predict(feat_df, "OT", "date")
                lgbm_preds = self.lgbm.predict(feat_df, "OT", "date")
                xgb_preds = self.xgboost.predict(feat_df, "OT", "date")
            except Exception:
                ridge_preds = arima_preds.copy()
                lgbm_preds = arima_preds.copy()
                xgb_preds = arima_preds.copy()

        preds_dict = {
            "arima": arima_preds[:steps],
            "ridge": ridge_preds[:steps],
            "lgbm": lgbm_preds[:steps],
            "xgboost": xgb_preds[:steps],
        }
        ensemble_preds = self.ensemble.predict(preds_dict)

        result: dict = {
            "forecast": [],
            "ensemble_weights": self.ensemble.get_weights(),
            "history_mode": history_mode,
        }
        if use_real_history:
            result["history_length"] = len(history)

        for i, dt_val in enumerate(future_dates):
            point: dict = {
                "datetime": dt_val.isoformat(),
                "prediction": float(ensemble_preds[i]),
            }
            if include_components:
                point["arima"] = float(preds_dict["arima"][i])
                point["ridge"] = float(preds_dict["ridge"][i])
                point["lgbm"] = float(preds_dict["lgbm"][i])
                point["xgboost"] = float(preds_dict["xgboost"][i])
            result["forecast"].append(point)

        return result


predictor = ForecastPredictor()
