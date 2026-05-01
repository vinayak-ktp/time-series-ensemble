import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class ExtraTreesForecaster(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = None,
        min_samples_leaf: int = 4,
        max_features: float = 0.5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_ = None
        self.scaler_ = StandardScaler()
        self._feature_cols = None

    def _get_feature_cols(self, df: pd.DataFrame, target_col: str, datetime_col: str) -> list:
        return [c for c in df.columns if c not in (target_col, datetime_col)]

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str,
        datetime_col: str,
    ) -> "ExtraTreesForecaster":
        self._feature_cols = self._get_feature_cols(train_df, target_col, datetime_col)
        X_train = train_df[self._feature_cols].values
        y_train = train_df[target_col].values

        X_scaled = self.scaler_.fit_transform(X_train)

        self.model_ = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X_scaled, y_train)
        return self

    def predict(self, df: pd.DataFrame, target_col: str, datetime_col: str) -> np.ndarray:
        X = df[self._feature_cols].values
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def get_params(self, deep=True) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
        }
