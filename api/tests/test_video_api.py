import json

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.schemas import AnalysisSource, AnalyzeRequest, AnalyzeResponse
from app.video_analysis import VideoBuildResult, build_motionmetrix_comparison


def _request() -> AnalyzeRequest:
    sequence = [[0.0] * 17 for _ in range(21)]
    return AnalyzeRequest.model_validate(
        {
            "session_id": "VIDEO_TEST",
            "patient_meta": {
                "patient_id": "TEST001",
                "height_cm": 175,
                "weight_kg": 70,
                "running_speed_kmh": 10,
            },
            "overstride_features": {
                "os_clip_selected_mm": 70.0,
                "os_event_abs_mean_mm": 68.0,
                "knee_landing_event_median_deg": 18.0,
                "thigh_flexion_recalc_deg": 30.0,
                "hip_rom_recalc_deg": 48.0,
            },
            "strike_feet": [
                {"foot": "left", "events": [{"sequence": sequence}]},
                {"foot": "right", "events": [{"sequence": sequence}]},
            ],
            "posture_metrics": {
                "cadence_spm": 170.0,
                "hip_rom_deg": 58.0,
                "max_knee_flexion_swing_deg": 94.0,
                "vertical_displacement_mm": 72.0,
            },
            "quality": {"quality_tier": "medium", "side_pose_rate": 0.98},
        }
    )


def _response() -> AnalyzeResponse:
    return AnalyzeResponse.model_validate(
        {
            "session_id": "VIDEO_TEST",
            "patient_id": "TEST001",
            "overstride": {
                "session_id": "VIDEO_TEST",
                "patient_id": "TEST001",
                "prediction_mm": 71.0,
                "prediction_std_mm": 4.0,
                "model_pair_count": 30,
                "quality_tier": "medium",
                "review_required": False,
                "status": "completed",
                "reason": None,
                "model_version": "final-20260825",
            },
            "strike_type": {
                "session_id": "VIDEO_TEST",
                "patient_id": "TEST001",
                "status": "completed",
                "reason": None,
                "feet": [
                    {"foot": "left", "prediction": "heel", "confidence": 0.9, "review_required": False},
                    {"foot": "right", "prediction": "heel", "confidence": 0.9, "review_required": False},
                ],
                "patient_anchor_class": "heel",
                "patient_anchor_confidence": 0.9,
                "final_class": "heel",
                "final_confidence": 0.9,
                "review_required": False,
                "rescue_activated": False,
                "model_version": "final-20260825",
            },
            "posture_metrics": _request().posture_metrics.model_dump(),
            "quality": {"quality_tier": "medium", "side_pose_rate": 0.98, "status": "completed"},
            "model_info": {"model_version": "final-20260825"},
        }
    )


def test_video_endpoint_requires_token(monkeypatch):
    monkeypatch.setenv("RUNNINGAI_API_TOKEN", "video-token")
    c = TestClient(app)
    r = c.post(
        "/api/v1/analyze/video",
        data={"patient_meta": json.dumps({"patient_id": "TEST001"})},
        files={"side_video": ("side.mp4", b"not-a-real-video", "video/mp4")},
    )
    assert r.status_code == 401


def test_video_endpoint_contract(monkeypatch):
    monkeypatch.setenv("RUNNINGAI_API_TOKEN", "video-token")

    async def fake_builder(**kwargs):
        return VideoBuildResult(
            request=_request(),
            source=AnalysisSource(
                input_type="raw_video",
                side_video_filename="side.mp4",
                analysis_start_sec=0.0,
                analysis_duration_sec=12.0,
                side_video_fps=60.0,
            ),
            motionmetrix={"overstride_mean_mm": 70.0, "hip_rom_mean_deg": 60.0},
        )

    monkeypatch.setattr(main_module, "build_analyze_request_from_videos", fake_builder)
    monkeypatch.setattr(main_module, "analyze", lambda req: _response())

    c = TestClient(app)
    r = c.post(
        "/api/v1/analyze/video",
        headers={"Authorization": "Bearer video-token"},
        data={
            "patient_meta": json.dumps({"patient_id": "TEST001", "height_cm": 175}),
            "motionmetrix": json.dumps({"overstride_mean_mm": 70.0, "hip_rom_mean_deg": 60.0}),
            "analysis_duration_sec": "12",
        },
        files={"side_video": ("side.mp4", b"fixture", "video/mp4")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"]["input_type"] == "raw_video"
    assert body["motionmetrix_comparison"]["overstride"]["difference"] == 1.0
    assert body["motionmetrix_comparison"]["hip_rom"]["difference"] == -2.0
    assert body["posture_metrics"]["max_knee_flexion_swing_deg"] == 94.0
    assert body["posture_metrics"]["vertical_displacement_mm"] == 72.0


def test_motionmetrix_comparison_uses_available_targets():
    analysis = _response().model_dump()
    result = build_motionmetrix_comparison(
        analysis,
        {
            "overstride_mean_mm": 70,
            "cadence_steps_per_min": 172,
            "left_knee_flexion_swing_max_deg": 93,
            "right_knee_flexion_swing_max_deg": 95,
            "vertical_oscillation_mean_mm": 70,
        },
    )
    assert result["overstride"]["difference"] == 1.0
    assert result["cadence"]["difference"] == -2.0
    assert result["max_knee_flexion_swing"]["motionmetrix_value"] == 94.0
    assert result["vertical_displacement"]["difference"] == 2.0
