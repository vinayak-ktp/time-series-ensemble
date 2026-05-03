from sklearn.ensemble import ExtraTreesRegressor


class ExtraTreesForecaster:
    def __init__(
        self,
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.5,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.model_ = None
        self._feature_cols = None

    def fit(
        self,
        train_feat,
        target_col,
        datetime_col,
    ):
        feature_cols = [
            c for c in train_feat.columns if c not in [target_col, datetime_col]
        ]
        self._feature_cols = feature_cols

        X_train = train_feat[feature_cols].values
        y_train = train_feat[target_col].values

        self.model_ = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, df):
        X = df[self._feature_cols].values
        return self.model_.predict(X)

    def get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
        }
