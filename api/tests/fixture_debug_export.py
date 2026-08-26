from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd


def build_debug_export_bytes() -> bytes:
    fps = 60.0
    t = np.arange(0.0, 2.51, 1.0 / fps)
    phase = 2.0 * np.pi * 2.0 * t
    frames = pd.DataFrame({
        "timestamp_sec": t,
        "pose_detected": True,
        "pelvis_center_x": 320.0 + 3.0 * np.sin(phase),
        "pelvis_center_y": 220.0 + 4.0 * np.sin(phase),
        "ground_y_px": 430.0,
        "forward_lean_deg": 4.0 + 0.3 * np.sin(phase),
        "left_heel_x": 270.0 + 15.0 * np.sin(phase),
        "left_heel_y": 415.0 + 3.0 * np.sin(phase),
        "left_toe_x": 315.0 + 15.0 * np.sin(phase),
        "left_toe_y": 420.0 + 2.0 * np.sin(phase),
        "left_ankle_x": 286.0 + 15.0 * np.sin(phase),
        "left_ankle_y": 385.0 + 3.0 * np.sin(phase),
        "left_shank_angle_deg": 7.0 + 1.0 * np.sin(phase),
        "left_knee_flexion_deg": 20.0 + 4.0 * np.sin(phase),
        "left_thigh_angle_deg": 12.0 + 18.0 * np.sin(phase),
        "right_heel_x": 360.0 - 15.0 * np.sin(phase),
        "right_heel_y": 414.0 - 3.0 * np.sin(phase),
        "right_toe_x": 405.0 - 15.0 * np.sin(phase),
        "right_toe_y": 419.0 - 2.0 * np.sin(phase),
        "right_ankle_x": 376.0 - 15.0 * np.sin(phase),
        "right_ankle_y": 384.0 - 3.0 * np.sin(phase),
        "right_shank_angle_deg": 7.5 - 1.0 * np.sin(phase),
        "right_knee_flexion_deg": 21.0 - 4.0 * np.sin(phase),
        "right_thigh_angle_deg": 11.0 - 18.0 * np.sin(phase),
    })

    event_rows = []
    for foot, times in [("left", [0.40, 1.00, 1.60, 2.20]), ("right", [0.70, 1.30, 1.90])]:
        for i, ic in enumerate(times):
            event_rows.append({
                "event_id": f"{foot}_{i}",
                "foot": foot,
                "initial_contact_time_sec": ic,
                "toe_off_time_sec": ic + 0.24,
                "contact_time_ms": 256.6666667,
                "pelvis_to_landing_ankle_dx_mm_est": 70.0 + i * 2.0,
                "foot_angle_at_contact_deg": 9.0 if foot == "left" else 10.0,
                "shank_angle_at_contact_deg": 7.0 if foot == "left" else 8.0,
                "knee_flexion_at_contact_deg": 20.0 if foot == "left" else 21.0,
                "foot_strike_type_estimate": "heel_candidate",
                "source_fps": fps,
            })
    events = pd.DataFrame(event_rows).sort_values("initial_contact_time_sec")
    summary = pd.DataFrame([{
        "actual_video_fps": fps,
        "analysis_fps": fps,
        "valid_duration_sec": 2.3,
        "overstride_avg_mm_est": 73.0,
        "overstride_trimmed_mean_mm_est": 72.5,
        "overstride_selected_mm_est": 72.8,
    }])

    rear_frames = pd.DataFrame({
        "timestamp_sec": t,
        "pose_detected": True,
        "pelvis_center_x": 320.0,
        "pelvis_center_y": 220.0,
        "rear_pelvic_tilt_deg": 2.0 * np.sin(phase),
    })
    rear_events = events[["event_id", "foot", "initial_contact_time_sec", "toe_off_time_sec", "source_fps"]].copy()
    rear_summary = pd.DataFrame([{"actual_video_fps": fps, "analysis_fps": fps}])
    meta = {"patient_id": "test001", "session_id": "test001", "height_cm": 175, "weight_kg": 70, "running_speed_kmh": 9.0}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("session/session_meta.json", json.dumps(meta))
        zf.writestr("session/motionmetrix_values.json", "{}")
        zf.writestr("session/visual_labels.json", "{}")
        zf.writestr("processed/side_running/side_running_all_frame_metrics.csv", frames.to_csv(index=False))
        zf.writestr("processed/side_running/side_running_gait_events.csv", events.to_csv(index=False))
        zf.writestr("processed/side_running/side_running_clip_summary.csv", summary.to_csv(index=False))
        zf.writestr("processed/rear_running/rear_running_all_frame_metrics.csv", rear_frames.to_csv(index=False))
        zf.writestr("processed/rear_running/rear_running_gait_events.csv", rear_events.to_csv(index=False))
        zf.writestr("processed/rear_running/rear_running_clip_summary.csv", rear_summary.to_csv(index=False))
    return buf.getvalue()
