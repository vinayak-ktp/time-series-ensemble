import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RidgeForecaster:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model_ = None
        self.scaler_ = StandardScaler()
        self._feature_cols = None

    def _get_feature_cols(self, df: pd.DataFrame, target_col: str, datetime_col: str) -> list:
        return [
            c for c in df.columns
            if c not in (target_col, datetime_col)
        ]

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str,
        datetime_col: str,
    ) -> "RidgeForecaster":
        self._feature_cols = self._get_feature_cols(train_df, target_col, datetime_col)
        X_train = train_df[self._feature_cols].values
        y_train = train_df[target_col].values
        X_scaled = self.scaler_.fit_transform(X_train)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X_scaled, y_train)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        target_col: str,
        datetime_col: str,
    ) -> np.ndarray:
        X = df[self._feature_cols].values
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def get_params(self) -> dict:
        return {"alpha": self.alpha}
