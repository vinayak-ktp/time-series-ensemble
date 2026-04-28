"""
Prophet Model Wrapper for time series forecasting.
Facebook Prophet handles trend + multiple seasonalities + holidays.
"""
import numpy as np
import pandas as pd
from prophet import Prophet
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetForecaster:
    """Prophet-based forecaster."""

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        horizon: int = 24,
        freq: str = "H",
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.horizon = horizon
        self.freq = freq
        self.model_ = None
        self._last_ds = None

    def fit(self, df: pd.DataFrame, datetime_col: str, target_col: str) -> "ProphetForecaster":
        """Fit prophet on a DataFrame with datetime and target columns."""
        prophet_df = df[[datetime_col, target_col]].rename(
            columns={datetime_col: "ds", target_col: "y"}
        )
        self.model_ = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )
        self.model_.fit(prophet_df)
        self._last_ds = prophet_df["ds"].max()
        return self

    def predict(self, steps: int = None) -> np.ndarray:
        """Predict `steps` steps into the future from last training date."""
        n = steps or self.horizon
        future = self.model_.make_future_dataframe(periods=n, freq=self.freq)
        forecast = self.model_.predict(future)
        return forecast["yhat"].values[-n:]

    def predict_on_df(self, df: pd.DataFrame, datetime_col: str) -> np.ndarray:
        """Predict on a provided dataframe (for val/test)."""
        future = df[[datetime_col]].rename(columns={datetime_col: "ds"})
        forecast = self.model_.predict(future)
        return forecast["yhat"].values

    def get_params(self) -> dict:
        return {
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "seasonality_mode": self.seasonality_mode,
        }
