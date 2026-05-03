import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocessing import chronological_split  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_all_metrics,
    mae,
    r2,
    rmse,
    smape,
)
from src.features.engineering import (  # noqa: E402
    featurize,
    make_lag_features,
    make_rolling_features,
    make_time_features,
)


@pytest.fixture
def sample_df():
    n = 500
    dates = pd.date_range("2021-01-01", periods=n, freq="h")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "date": dates,
            "OT": np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 0.1 + 5,
            "HUFL": np.random.randn(n),
        }
    )


class TestChronologicalSplit:
    def test_split_sizes(self, sample_df):
        train, val, test = chronological_split(sample_df, test_size=0.2, val_size=0.1)
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_no_overlap(self, sample_df):
        train, val, test = chronological_split(sample_df, test_size=0.2, val_size=0.1)
        assert train["date"].max() < val["date"].min()
        assert val["date"].max() < test["date"].min()

    def test_correct_proportions(self, sample_df):
        n = len(sample_df)
        train, val, test = chronological_split(sample_df, test_size=0.2, val_size=0.1)
        assert len(test) == pytest.approx(int(n * 0.2), abs=2)
        assert len(val) == pytest.approx(int(n * 0.1), abs=2)


class TestLagFeatures:
    def test_creates_lag_columns(self, sample_df):
        df = make_lag_features(sample_df.copy(), "OT", [1, 2, 24])
        assert "OT_lag_1" in df.columns
        assert "OT_lag_24" in df.columns

    def test_lag_1_is_shifted(self, sample_df):
        df = make_lag_features(sample_df.copy(), "OT", [1])
        assert pd.isna(df["OT_lag_1"].iloc[0])
        assert df["OT_lag_1"].iloc[1] == df["OT"].iloc[0]


class TestRollingFeatures:
    def test_creates_rolling_columns(self, sample_df):
        df = make_rolling_features(sample_df.copy(), "OT", [6, 12])
        for w in [6, 12]:
            assert f"OT_roll_mean_{w}" in df.columns
            assert f"OT_roll_std_{w}" in df.columns

    def test_rolling_not_future_leak(self, sample_df):
        df = make_rolling_features(sample_df.copy(), "OT", [6])
        assert pd.isna(df["OT_roll_mean_6"].iloc[0])


class TestTimeFeatures:
    def test_creates_time_columns(self, sample_df):
        df = make_time_features(sample_df.copy(), "date")
        for col in ["hour", "dayofweek", "month", "hour_sin", "hour_cos"]:
            assert col in df.columns

    def test_hour_within_range(self, sample_df):
        df = make_time_features(sample_df.copy(), "date")
        assert df["hour"].between(0, 23).all()

    def test_cyclical_encoding_bounded(self, sample_df):
        df = make_time_features(sample_df.copy(), "date")
        assert df["hour_sin"].between(-1.0, 1.0).all()
        assert df["hour_cos"].between(-1.0, 1.0).all()


class TestFeaturize:
    def test_drops_nan_rows(self, sample_df):
        df_feat = featurize(sample_df, "OT", "date", [1, 168], [24], use_time_features=True)
        assert not df_feat.isnull().any().any()

    def test_feature_count_increases(self, sample_df):
        df_feat = featurize(sample_df, "OT", "date", [1, 2], [6], use_time_features=True)
        assert len(df_feat.columns) > len(sample_df.columns)


class TestMetrics:
    def setup_method(self):
        np.random.seed(0)
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    def test_mae_nonnegative(self):
        assert mae(self.y_true, self.y_pred) >= 0

    def test_rmse_ge_mae(self):
        assert rmse(self.y_true, self.y_pred) >= mae(self.y_true, self.y_pred)

    def test_perfect_prediction_zero_error(self):
        assert mae(self.y_true, self.y_true) < 1e-10
        assert rmse(self.y_true, self.y_true) < 1e-10

    def test_r2_perfect_is_one(self):
        assert abs(r2(self.y_true, self.y_true) - 1.0) < 1e-8

    def test_compute_all_returns_all_keys(self):
        m = compute_all_metrics(self.y_true, self.y_pred)
        for key in ["mae", "mse", "rmse", "mape", "smape", "r2"]:
            assert key in m

    def test_smape_bounded(self):
        val = smape(abs(self.y_true), abs(self.y_pred))
        assert 0 <= val <= 200
