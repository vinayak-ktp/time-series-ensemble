"""
Integration tests for the FastAPI endpoints.
Uses httpx.AsyncClient for async test support.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.main import app


@pytest.fixture
def client():
    """Test client that does NOT load real models (models not available in CI)."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_health_has_version(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_docs_link(self, client):
        resp = client.get("/")
        assert "/docs" in resp.json().get("docs", "")


class TestPredictEndpoint:
    def test_predict_unloaded_returns_503(self, client):
        """Without trained models, /predict should return 503."""
        payload = {"start_datetime": "2018-06-01T00:00:00", "steps": 5}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 503

    def test_predict_invalid_steps(self, client):
        """steps must be 1–720."""
        payload = {"start_datetime": "2018-06-01T00:00:00", "steps": 0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_too_many_steps(self, client):
        payload = {"start_datetime": "2018-06-01T00:00:00", "steps": 9999}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


class TestModelsEndpoint:
    def test_models_returns_200(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200

    def test_models_lists_four_models(self, client):
        resp = client.get("/models")
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) == 4
