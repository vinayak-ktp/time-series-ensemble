"""
Ensemble Model
Combines ARIMA, Prophet, LightGBM, and XGBoost predictions via:
  1. Simple average
  2. Weighted average (weights optimized on validation set)
  3. Stacking with a Ridge meta-model
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def _mse(weights: np.ndarray, preds: np.ndarray, y_true: np.ndarray) -> float:
    blended = np.dot(preds, weights)
    return mean_squared_error(y_true, blended)


class EnsembleForecaster:
    """
    Ensemble of base forecasters.

    Parameters
    ----------
    method : str
        'simple_average' | 'weighted_average' | 'stacking'
    """

    def __init__(self, method: str = "weighted_average"):
        self.method = method
        self.weights_ = None
        self.meta_model_ = None
        self.model_names_ = None

    # ------------------------------------------------------------------
    def fit_weights(
        self,
        val_preds: dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> "EnsembleForecaster":
        """Optimize ensemble weights on the validation set."""
        self.model_names_ = list(val_preds.keys())
        P = np.column_stack([val_preds[n] for n in self.model_names_])

        if self.method == "simple_average":
            n = len(self.model_names_)
            self.weights_ = np.ones(n) / n

        elif self.method == "weighted_average":
            n = len(self.model_names_)
            x0 = np.ones(n) / n
            bounds = [(0, 1)] * n
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
            result = minimize(
                _mse,
                x0,
                args=(P, y_val),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            self.weights_ = result.x
            print(f"[ensemble] Optimized weights: {dict(zip(self.model_names_, self.weights_.round(4)))}")

        elif self.method == "stacking":
            self.meta_model_ = Ridge(alpha=1.0)
            self.meta_model_.fit(P, y_val)
            self.weights_ = self.meta_model_.coef_

        return self

    def predict(self, test_preds: dict[str, np.ndarray]) -> np.ndarray:
        """Blend test predictions using fitted weights."""
        P = np.column_stack([test_preds[n] for n in self.model_names_])

        if self.method == "stacking" and self.meta_model_ is not None:
            return self.meta_model_.predict(P)
        else:
            return np.dot(P, self.weights_)

    def get_weights(self) -> dict:
        if self.model_names_ is None:
            return {}
        return dict(zip(self.model_names_, self.weights_.tolist()))
