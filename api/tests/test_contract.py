import os

from fastapi.testclient import TestClient

from app.main import app


def test_health_is_public():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["api_version"] == "v1.4.0"
    assert r.json()["model_version"] == "final-20260825"


def test_model_info_requires_token():
    c = TestClient(app)
    r = c.get("/api/v1/model/info")
    assert r.status_code == 401


def test_invalid_token_is_rejected(monkeypatch):
    monkeypatch.setenv("RUNNINGAI_API_TOKEN", "expected-token")
    c = TestClient(app)
    r = c.get(
        "/api/v1/model/info",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "AUTH_INVALID"


def test_openapi_declares_bearer_security():
    schema = app.openapi()
    scheme = schema["components"]["securitySchemes"]["BearerAuth"]
    assert scheme["type"] == "http"
    assert scheme["scheme"] == "bearer"
    operation = schema["paths"]["/api/v1/model/info"]["get"]
    assert {"BearerAuth": []} in operation["security"]
    video_operation = schema["paths"]["/api/v1/analyze/video"]["post"]
    assert {"BearerAuth": []} in video_operation["security"]
    assert "security" not in schema["paths"]["/health"]["get"]
