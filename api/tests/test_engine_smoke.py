import numpy as np

from app.main import get_engine


def test_artifact_contract_loads():
    engine = get_engine()
    assert engine.status.artifact_integrity is True
    assert len(engine.status.sequence_channels) == 17
    assert len(engine.status.time_grid_sec) == 21


def test_overstride_smoke():
    engine = get_engine()
    out = engine.predict_overstride(
        {
            "os_clip_selected_mm": 72.0,
            "os_event_abs_mean_mm": 70.0,
            "os_event_abs_iqr_mm": 8.0,
            "os_lr_abs_diff_mm": 5.0,
            "foot_angle_abs_median_deg": 12.0,
            "shank_angle_abs_median_deg": 7.0,
            "knee_landing_event_median_deg": 18.0,
            "thigh_flexion_recalc_deg": 30.0,
            "hip_rom_recalc_deg": 48.0,
            "running_speed_kmh": 9.0,
            "height_cm": 175.0,
            "side_pose_rate": 0.98,
        }
    )
    assert out["status"] == "completed"
    assert np.isfinite(out["prediction_mm"])
    assert out["model_pair_count"] == 30
