import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings("ignore")


class CatBoostForecaster(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        reg_lambda: float = 3.0,
        min_child_samples: int = 20,
        random_seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.random_seed = random_seed
        self.model_ = None
        self.feature_names_ = None

    def _get_feature_matrix(
        self, df: pd.DataFrame, target_col: str, datetime_col: str
    ) -> tuple[np.ndarray, np.ndarray, list]:
        drop_cols = [target_col, datetime_col] if datetime_col in df.columns else [target_col]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns]).values
        y = df[target_col].values
        feature_names = [c for c in df.columns if c not in drop_cols]
        return X, y, feature_names

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str,
        datetime_col: str,
    ) -> "CatBoostForecaster":
        X_train, y_train, feat_names = self._get_feature_matrix(train_df, target_col, datetime_col)
        X_val, y_val, _ = self._get_feature_matrix(val_df, target_col, datetime_col)
        self.feature_names_ = feat_names

        self.model_ = CatBoostRegressor(
            iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            depth=self.max_depth,
            subsample=self.subsample,
            l2_leaf_reg=self.reg_lambda,
            min_child_samples=self.min_child_samples,
            random_seed=self.random_seed,
            loss_function="RMSE",
            eval_metric="RMSE",
            early_stopping_rounds=50,
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )
        return self

    def predict(self, df: pd.DataFrame, target_col: str, datetime_col: str) -> np.ndarray:
        drop_cols = [target_col, datetime_col] if datetime_col in df.columns else [target_col]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns]).values
        return self.model_.predict(X)

    def get_params(self, deep=True) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "reg_lambda": self.reg_lambda,
        }
