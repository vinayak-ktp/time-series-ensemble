import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.main import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_has_docs_link(self, client):
        assert "/docs" in client.get("/").json().get("docs", "")


class TestPredictEndpoint:
    def test_predict_unloaded_returns_503(self, client):
        from api import predictor as pred_module

        original = pred_module.predictor._loaded
        pred_module.predictor._loaded = False
        try:
            resp = client.post(
                "/predict", json={"start_datetime": "2018-06-01T00:00:00", "steps": 5}
            )
            assert resp.status_code == 503
        finally:
            pred_module.predictor._loaded = original

    def test_predict_invalid_steps(self, client):
        resp = client.post("/predict", json={"start_datetime": "2018-06-01T00:00:00", "steps": 0})
        assert resp.status_code == 422

    def test_predict_too_many_steps(self, client):
        resp = client.post(
            "/predict", json={"start_datetime": "2018-06-01T00:00:00", "steps": 9999}
        )
        assert resp.status_code == 422


class TestModelsEndpoint:
    def test_models_returns_200(self, client):
        assert client.get("/models").status_code == 200

    def test_models_lists_all_trained_models(self, client):
        data = client.get("/models").json()
        assert "all_trained_models" in data
        assert len(data["all_trained_models"]) == 5

    def test_models_lists_hybrid_components(self, client):
        data = client.get("/models").json()
        assert "hybrid_base" in data
        assert "hybrid_residuals" in data
        assert data["hybrid_base"] == "ridge"
        assert len(data["hybrid_residuals"]) == 2
