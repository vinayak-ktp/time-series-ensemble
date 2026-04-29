import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings("ignore")


class LGBMForecaster(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
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
    ) -> "LGBMForecaster":
        X_train, y_train, feat_names = self._get_feature_matrix(train_df, target_col, datetime_col)
        X_val, y_val, _ = self._get_feature_matrix(val_df, target_col, datetime_col)
        self.feature_names_ = feat_names

        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            verbose=-1,
        )
        self.model_.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
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
            "num_leaves": self.num_leaves,
        }
