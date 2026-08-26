import os

import numpy as np
from fastapi.testclient import TestClient

os.environ["RUNNINGAI_API_TOKEN"] = "test-token"

from app.debug_export_adapter import DebugExportAdapter
from app.main import analyze, app, get_engine
from tests.fixture_debug_export import build_debug_export_bytes


def test_debug_export_adapter_http_parity():
    data = build_debug_export_bytes()
    result = DebugExportAdapter.from_bytes(data).build()
    request = result.request.model_dump(mode="json")
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-token"}

    adapted = client.post(
        "/api/v1/adapter/debug-export",
        headers=headers,
        files={"file": ("test.zip", data, "application/zip")},
    )
    assert adapted.status_code == 200
    assert adapted.json() == request


def test_direct_engine_api_parity(monkeypatch):
    data = build_debug_export_bytes()
    result = DebugExportAdapter.from_bytes(data).build()
    request = result.request.model_dump(mode="json")
    engine = get_engine()
    direct_o = engine.predict_overstride(request["overstride_features"])
    direct_s = engine.predict_strike(request["strike_feet"])

    class CachedEngine:
        status = engine.status

        def predict_overstride(self, features):
            return direct_o

        def predict_strike(self, feet):
            return direct_s

    import app.main as main

    monkeypatch.setattr(main, "get_engine", lambda: CachedEngine())
    body = analyze(result.request).model_dump(mode="json")
    assert np.isclose(body["overstride"]["prediction_mm"], direct_o["prediction_mm"], atol=1e-12, rtol=0)
    assert np.isclose(body["overstride"]["prediction_std_mm"], direct_o["prediction_std_mm"], atol=1e-12, rtol=0)
    assert body["overstride"]["model_pair_count"] == direct_o["model_pair_count"]
    assert body["strike_type"]["patient_anchor_class"] == direct_s["patient_anchor_class"]
    assert np.isclose(body["strike_type"]["patient_anchor_confidence"], direct_s["patient_anchor_confidence"], atol=1e-12, rtol=0)
    assert np.isclose(body["strike_type"]["rescue_p_heel"], direct_s["rescue_p_heel"], atol=1e-12, rtol=0)
    assert body["strike_type"]["rescue_activated"] == direct_s["rescue_activated"]
    assert body["strike_type"]["final_class"] == direct_s["final_class"]
    assert np.isclose(body["strike_type"]["final_confidence"], direct_s["final_confidence"], atol=1e-12, rtol=0)
    assert body["strike_type"]["review_required"] == direct_s["review_required"]
    assert body["strike_type"]["feet"] == direct_s["feet"]
    for foot in body["strike_type"]["feet"]:
        assert "prediction" in foot
        assert "confidence" in foot
        assert "local_probabilities" in foot
