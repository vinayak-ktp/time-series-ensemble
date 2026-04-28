"""
Tests for the Ensemble model.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.ensemble import EnsembleForecaster


@pytest.fixture
def val_preds():
    np.random.seed(42)
    n = 200
    return {
        "arima":   np.random.randn(n) * 0.5 + 3,
        "prophet": np.random.randn(n) * 0.5 + 3,
        "lgbm":    np.random.randn(n) * 0.5 + 3,
        "xgboost": np.random.randn(n) * 0.5 + 3,
    }


@pytest.fixture
def y_val(val_preds):
    np.random.seed(0)
    return np.random.randn(200) * 0.3 + 3


class TestSimpleAverage:
    def test_simple_average_equal_weights(self, val_preds, y_val):
        ens = EnsembleForecaster(method="simple_average")
        ens.fit_weights(val_preds, y_val)
        w = ens.get_weights()
        for v in w.values():
            assert abs(v - 0.25) < 1e-8

    def test_predict_shape(self, val_preds, y_val):
        ens = EnsembleForecaster(method="simple_average")
        ens.fit_weights(val_preds, y_val)
        preds = ens.predict(val_preds)
        assert preds.shape == (200,)


class TestWeightedAverage:
    def test_weights_sum_to_one(self, val_preds, y_val):
        ens = EnsembleForecaster(method="weighted_average")
        ens.fit_weights(val_preds, y_val)
        assert abs(sum(ens.get_weights().values()) - 1.0) < 1e-5

    def test_weights_nonnegative(self, val_preds, y_val):
        ens = EnsembleForecaster(method="weighted_average")
        ens.fit_weights(val_preds, y_val)
        for v in ens.get_weights().values():
            assert v >= -1e-8


class TestStacking:
    def test_stacking_runs(self, val_preds, y_val):
        ens = EnsembleForecaster(method="stacking")
        ens.fit_weights(val_preds, y_val)
        preds = ens.predict(val_preds)
        assert preds.shape == (200,)
