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
        self._train_values = None

    def fit(self, y: pd.Series) -> "ARIMAForecaster":
        self._train_values = y.values.copy()
        self.model_ = ARIMA(self._train_values, order=(self.p, self.d, self.q))
        self.result_ = self.model_.fit()
        return self

    def predict(self, steps: int = None) -> np.ndarray:
        n = steps or self.horizon
        return np.array(self.result_.forecast(steps=n))

    def rolling_forecast(self, y_test: np.ndarray) -> np.ndarray:
        history = list(self._train_values)
        predictions = []
        h = self.horizon
        n = len(y_test)
        i = 0
        while i < n:
            window = min(h, n - i)
            model = ARIMA(history, order=(self.p, self.d, self.q))
            result = model.fit()
            preds = result.forecast(steps=window)
            predictions.extend(preds[:window])
            history.extend(y_test[i : i + window])
            i += window
        return np.array(predictions[:n])

    def predict_in_sample(self, start: int, end: int) -> np.ndarray:
        return np.array(self.result_.predict(start=start, end=end))

    def get_params(self) -> dict:
        return {"p": self.p, "d": self.d, "q": self.q, "horizon": self.horizon}
