import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


class ARIMAForecaster:
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
        return self

    def predict(self, steps: int = None) -> np.ndarray:
        n = steps or self.horizon
        return np.array(self.result_.forecast(steps=n))

    def predict_in_sample(self, start: int, end: int) -> np.ndarray:
        return np.array(self.result_.predict(start=start, end=end))

    def get_params(self) -> dict:
        return {"p": self.p, "d": self.d, "q": self.q, "horizon": self.horizon}
