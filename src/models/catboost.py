from catboost import CatBoostRegressor


class CatBoostForecaster:
    def __init__(
        self,
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        reg_lambda=1.0,
        min_child_samples=10,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.model_ = None
        self._feature_cols = None

    def fit(
        self,
        train_feat,
        val_feat,
        target_col,
        datetime_col,
    ):
        feature_cols = [
            c for c in train_feat.columns if c not in [target_col, datetime_col]
        ]
        self._feature_cols = feature_cols

        X_train = train_feat[feature_cols].values
        y_train = train_feat[target_col].values
        X_val = val_feat[feature_cols].values
        y_val = val_feat[target_col].values

        self.model_ = CatBoostRegressor(
            iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            depth=self.max_depth,
            subsample=self.subsample,
            l2_leaf_reg=self.reg_lambda,
            min_data_in_leaf=self.min_child_samples,
            random_seed=self.random_state,
            verbose=False,
            early_stopping_rounds=50,
        )
        self.model_.fit(X_train, y_train, eval_set=(X_val, y_val))
        return self

    def predict(self, df, target_col, datetime_col):
        X = df[self._feature_cols].values
        return self.model_.predict(X)

    def get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "reg_lambda": self.reg_lambda,
            "min_child_samples": self.min_child_samples,
        }
