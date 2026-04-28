import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings("ignore")


class XGBForecaster(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_weight: int = 5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.random_state = random_state
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
    ) -> "XGBForecaster":
        X_train, y_train, feat_names = self._get_feature_matrix(train_df, target_col, datetime_col)
        X_val, y_val, _ = self._get_feature_matrix(val_df, target_col, datetime_col)
        self.feature_names_ = feat_names

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)

        params = {
            "objective": "reg:squarederror",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "seed": self.random_state,
            "verbosity": 0,
        }
        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        return self

    def predict(self, df: pd.DataFrame, target_col: str, datetime_col: str) -> np.ndarray:
        drop_cols = [target_col, datetime_col] if datetime_col in df.columns else [target_col]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns]).values
        feat_names = [c for c in df.columns if c not in drop_cols]
        dtest = xgb.DMatrix(X, feature_names=feat_names)
        return self.model_.predict(dtest)

    def get_params(self, deep=True) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
        }
