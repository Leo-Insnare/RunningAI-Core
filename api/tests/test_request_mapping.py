from fastapi.testclient import TestClient

import app.main as main


class StubEngine:
    def __init__(self):
        self.features = None

    def predict_overstride(self, features):
        self.features = dict(features)
        return {
            "status": "completed",
            "reason": None,
            "prediction_mm": 70.0,
            "prediction_std_mm": 1.0,
            "model_pair_count": 30,
        }


def test_overstride_patient_meta_is_merged(monkeypatch):
    stub = StubEngine()
    monkeypatch.setenv("RUNNINGAI_API_TOKEN", "test-token")
    monkeypatch.setattr(main, "get_engine", lambda: stub)
    client = TestClient(main.app)
    payload = {
        "session_id": "s1",
        "patient_meta": {
            "patient_id": "p1",
            "height_cm": 181.0,
            "running_speed_kmh": 10.5
        },
        "features": {
            "os_clip_selected_mm": 72.0,
            "os_event_abs_mean_mm": 70.0,
            "height_cm": 150.0,
            "running_speed_kmh": 5.0
        }
    }
    r = client.post(
        "/api/v1/predict/overstride",
        json=payload,
        headers={"Authorization": "Bearer test-token"},
    )
    assert r.status_code == 200
    assert stub.features["height_cm"] == 181.0
    assert stub.features["running_speed_kmh"] == 10.5
