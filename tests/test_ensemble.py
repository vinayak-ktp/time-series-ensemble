import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.ensemble import EnsembleForecaster  # noqa: E402


@pytest.fixture
def val_preds():
    np.random.seed(42)
    n = 200
    return {
        "ridge": np.random.randn(n) * 0.3 + 3,
        "catboost": np.random.randn(n) * 0.5 + 3,
        "extra_trees": np.random.randn(n) * 0.5 + 3,
    }


@pytest.fixture
def y_val():
    np.random.seed(0)
    return np.random.randn(200) * 0.3 + 3


class TestSimpleAverage:
    def test_simple_average_equal_weights(self, val_preds, y_val):
        ens = EnsembleForecaster(method="simple_average")
        ens.fit_weights(val_preds, y_val)
        n = len(val_preds)
        for v in ens.get_weights().values():
            assert abs(v - 1.0 / n) < 1e-8

    def test_predict_shape(self, val_preds, y_val):
        ens = EnsembleForecaster(method="simple_average")
        ens.fit_weights(val_preds, y_val)
        assert ens.predict(val_preds).shape == (200,)


class TestWeightedAverage:
    def test_weights_sum_to_one(self, val_preds, y_val):
        ens = EnsembleForecaster(method="weighted_average")
        ens.fit_weights(val_preds, y_val)
        assert abs(sum(ens.get_weights().values()) - 1.0) < 1e-5

    def test_weights_respect_min_floor(self, val_preds, y_val):
        min_w = 0.10
        ens = EnsembleForecaster(method="weighted_average", min_weight=min_w)
        ens.fit_weights(val_preds, y_val)
        for name, v in ens.get_weights().items():
            assert v >= min_w - 1e-6, f"{name} weight {v:.4f} below min_weight {min_w}"

    def test_weights_nonnegative(self, val_preds, y_val):
        ens = EnsembleForecaster(method="weighted_average")
        ens.fit_weights(val_preds, y_val)
        for v in ens.get_weights().values():
            assert v >= -1e-8


class TestStacking:
    def test_stacking_runs(self, val_preds, y_val):
        ens = EnsembleForecaster(method="stacking")
        ens.fit_weights(val_preds, y_val)
        assert ens.predict(val_preds).shape == (200,)
