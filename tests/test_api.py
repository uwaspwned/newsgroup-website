from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Config


def model_artifacts_exist() -> bool:
    return all([
        Config.MODEL_PATH.exists(),
        Path(f"{Config.MODEL_PATH}.sig").exists(),
        Config.PUBLIC_KEY_PATH.exists(),
    ])


@pytest.fixture(scope="module")
def main_module():
    if not model_artifacts_exist():
        pytest.skip("Model artifacts are not generated")

    import app.main as module

    return module


@pytest.fixture()
def client(main_module, monkeypatch):
    monkeypatch.setattr(Config, "VALID_API_KEYS", ["test-api-key"])
    monkeypatch.setattr(Config, "ALLOWED_IPS", [*Config.ALLOWED_IPS, "testclient"])
    return TestClient(main_module.app)


def test_text_request_sanitizes_whitespace_and_control_chars(main_module):
    request = main_module.TextPredictionRequest(text="  Space\n\t travel\x00\x07  ")

    assert request.text == "Space travel"


def test_text_request_rejects_blank_text(main_module):
    with pytest.raises(ValidationError):
        main_module.TextPredictionRequest(text="     ")


def test_text_request_rejects_too_short_text(main_module):
    with pytest.raises(ValidationError):
        main_module.TextPredictionRequest(text="ab")


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_requires_api_key(client):
    response = client.post("/predict", json={"text": "Space travel"})

    assert response.status_code == 403


def test_predict_endpoint_returns_category(client):
    response = client.post(
        "/predict",
        json={"text": "Space shuttle launched from Kennedy Space Center"},
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"text_class", "category_name", "text"}
    assert isinstance(body["text_class"], int)
    assert isinstance(body["category_name"], str)
    assert body["text"] == "Space shuttle launched from Kennedy Space Center"


def test_predict_endpoint_rejects_invalid_text(client):
    response = client.post(
        "/predict",
        json={"text": "   "},
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 422


def test_cache_info_endpoint(client):
    response = client.get("/cache-info")

    assert response.status_code == 200
    body = response.json()
    assert "enabled" in body
    assert "connected" in body
    assert "ttl_seconds" in body
    assert "cache_namespace" in body
