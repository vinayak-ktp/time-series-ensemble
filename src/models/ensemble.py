import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def _mse(weights: np.ndarray, preds: np.ndarray, y_true: np.ndarray) -> float:
    blended = np.dot(preds, weights)
    return mean_squared_error(y_true, blended)


class EnsembleForecaster:
    def __init__(self, method: str = "weighted_average", min_weight: float = 0.1):
        self.method = method
        self.min_weight = min_weight
        self.weights_ = None
        self.meta_model_ = None
        self.model_names_ = None

    def fit_weights(
        self,
        val_preds: dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> "EnsembleForecaster":
        self.model_names_ = list(val_preds.keys())
        P = np.column_stack([val_preds[n] for n in self.model_names_])

        if self.method == "simple_average":
            n = len(self.model_names_)
            self.weights_ = np.ones(n) / n

        elif self.method == "weighted_average":
            n = len(self.model_names_)
            x0 = np.ones(n) / n
            # Each model gets at least min_weight, remainder distributed freely.
            # Constraint: sum == 1. Bounds ensure min contribution per member.
            low = self.min_weight
            high = 1.0 - low * (n - 1)   # max any single model can receive
            bounds = [(low, high)] * n
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
            print(
                f"[ensemble] Optimized weights (min={low}): "
                f"{dict(zip(self.model_names_, self.weights_.round(4)))}"
            )

        elif self.method == "stacking":
            self.meta_model_ = Ridge(alpha=1.0)
            self.meta_model_.fit(P, y_val)
            self.weights_ = self.meta_model_.coef_

        return self

    def predict(self, test_preds: dict[str, np.ndarray]) -> np.ndarray:
        P = np.column_stack([test_preds[n] for n in self.model_names_])
        if self.method == "stacking" and self.meta_model_ is not None:
            return self.meta_model_.predict(P)
        return np.dot(P, self.weights_)

    def get_weights(self) -> dict:
        if self.model_names_ is None:
            return {}
        return dict(zip(self.model_names_, self.weights_.tolist()))
