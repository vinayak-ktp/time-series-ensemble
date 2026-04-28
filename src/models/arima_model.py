"""
ARIMA Model Wrapper for time series forecasting.
Uses statsmodels ARIMA with configurable (p, d, q) order.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


class ARIMAForecaster:
    """ARIMA forecaster that predicts `horizon` steps ahead via recursive forecasting."""

    def __init__(self, p: int = 2, d: int = 1, q: int = 2, horizon: int = 24):
        self.p = p
        self.d = d
        self.q = q
        self.horizon = horizon
        self.model_ = None
        self.result_ = None

    def fit(self, y: pd.Series) -> "ARIMAForecaster":
        self.model_ = ARIMA(y.values, order=(self.p, self.d, self.q))
        self.result_ = self.model_.fit()
        self._last_series = y.values
        return self

    def predict(self, steps: int = None) -> np.ndarray:
        """Forecast `steps` (default: horizon) steps ahead."""
        n = steps or self.horizon
        forecast = self.result_.forecast(steps=n)
        return np.array(forecast)

    def predict_in_sample(self, start: int, end: int) -> np.ndarray:
        """Return in-sample predictions for evaluation purposes."""
        preds = self.result_.predict(start=start, end=end)
        return np.array(preds)

    def get_params(self) -> dict:
        return {"p": self.p, "d": self.d, "q": self.q, "horizon": self.horizon}
