"""
Model predictor for the API.
Loads all model artifacts from disk and coordinates ensemble inference.
"""
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


MODELS_DIR = os.getenv("MODELS_DIR", "models")


class ForecastPredictor:
    """Loads trained models and runs ensemble forecasting."""

    def __init__(self):
        self.arima = None
        self.prophet = None
        self.lgbm = None
        self.xgboost = None
        self.ensemble = None
        self.scaler = None
        self._loaded = False

    def load(self) -> None:
        """Load all model pickles from MODELS_DIR."""
        try:
            with open(os.path.join(MODELS_DIR, "arima.pkl"), "rb") as f:
                self.arima = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "prophet.pkl"), "rb") as f:
                self.prophet = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "lgbm.pkl"), "rb") as f:
                self.lgbm = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
                self.xgboost = pickle.load(f)
            with open(os.path.join(MODELS_DIR, "ensemble.pkl"), "rb") as f:
                self.ensemble = pickle.load(f)
            if os.path.exists(os.path.join(MODELS_DIR, "scaler.pkl")):
                with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
                    self.scaler = pickle.load(f)
            self._loaded = True
            print("[predictor] All models loaded successfully.")
        except FileNotFoundError as e:
            print(f"[predictor] Warning: {e}. Run training pipeline first.")
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(
        self,
        start_datetime: str,
        steps: int = 24,
        include_components: bool = False,
    ) -> dict:
        """Generate ensemble forecast for `steps` hours starting from `start_datetime`."""
        if not self._loaded:
            raise RuntimeError("Models not loaded. Run the training pipeline first.")

        # Generate future datetime index
        start_dt = datetime.fromisoformat(start_datetime)
        future_dates = [start_dt + timedelta(hours=i) for i in range(steps)]
        future_df = pd.DataFrame({"date": future_dates})

        # ── Individual model forecasts ────────────────────────────────────
        arima_preds = self.arima.predict(steps=steps)
        prophet_preds = self.prophet.predict_on_df(future_df, "date")

        # For LGBM/XGBoost we need features - use prophet predictions as proxy target
        # (In production these would use real features from a feature store)
        # Build minimal feature dataframe from datetime features
        feat_df = future_df.copy()
        feat_df["OT"] = prophet_preds  # use prophet as target placeholder
        dt = pd.to_datetime(feat_df["date"])
        feat_df["hour"] = dt.dt.hour
        feat_df["dayofweek"] = dt.dt.dayofweek
        feat_df["month"] = dt.dt.month
        feat_df["dayofyear"] = dt.dt.dayofyear
        feat_df["week"] = dt.dt.isocalendar().week.astype(int)
        feat_df["quarter"] = dt.dt.quarter
        feat_df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df["hour"] / 24)
        feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df["hour"] / 24)
        feat_df["dow_sin"] = np.sin(2 * np.pi * feat_df["dayofweek"] / 7)
        feat_df["dow_cos"] = np.cos(2 * np.pi * feat_df["dayofweek"] / 7)
        feat_df["month_sin"] = np.sin(2 * np.pi * feat_df["month"] / 12)
        feat_df["month_cos"] = np.cos(2 * np.pi * feat_df["month"] / 12)
        # Add lag placeholders (use prophet preds shifted)
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            feat_df[f"OT_lag_{lag}"] = np.roll(prophet_preds, lag)
        for w in [6, 12, 24, 168]:
            feat_df[f"OT_roll_mean_{w}"] = pd.Series(prophet_preds).rolling(w, min_periods=1).mean().values
            feat_df[f"OT_roll_std_{w}"] = pd.Series(prophet_preds).rolling(w, min_periods=1).std().fillna(0).values
            feat_df[f"OT_roll_min_{w}"] = pd.Series(prophet_preds).rolling(w, min_periods=1).min().values
            feat_df[f"OT_roll_max_{w}"] = pd.Series(prophet_preds).rolling(w, min_periods=1).max().values

        # Add MULL columns that might exist in trained feature set
        for col in ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]:
            if col not in feat_df.columns:
                feat_df[col] = 0.0

        try:
            lgbm_preds = self.lgbm.predict(feat_df, "OT", "date")
            xgb_preds = self.xgboost.predict(feat_df, "OT", "date")
        except Exception:
            lgbm_preds = prophet_preds.copy()
            xgb_preds = prophet_preds.copy()

        preds_dict = {
            "arima": arima_preds[: steps],
            "prophet": prophet_preds[: steps],
            "lgbm": lgbm_preds[: steps],
            "xgboost": xgb_preds[: steps],
        }
        ensemble_preds = self.ensemble.predict(preds_dict)

        result = {
            "forecast": [],
            "ensemble_weights": self.ensemble.get_weights(),
        }
        for i, dt_val in enumerate(future_dates):
            point = {"datetime": dt_val.isoformat(), "prediction": float(ensemble_preds[i])}
            if include_components:
                point["arima"] = float(preds_dict["arima"][i])
                point["prophet"] = float(preds_dict["prophet"][i])
                point["lgbm"] = float(preds_dict["lgbm"][i])
                point["xgboost"] = float(preds_dict["xgboost"][i])
            result["forecast"].append(point)

        return result


# Singleton predictor instance
predictor = ForecastPredictor()
