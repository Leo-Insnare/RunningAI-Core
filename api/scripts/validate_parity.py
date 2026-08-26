from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import requests
from fastapi.testclient import TestClient

from app.debug_export_adapter import DebugExportAdapter
from app.main import analyze, app, get_engine

REFERENCE_ALIASES = {
    "cadence_same_foot_spm": ["cadence_same_foot_spm", "same_foot_cycle_spm"],
    "contact_time_inclusive_mean_ms": ["contact_time_inclusive_mean_ms", "contact_time_recalc_inclusive_mean_ms"],
    "forward_lean_recalc_deg": ["forward_lean_recalc_deg"],
    "thigh_flexion_recalc_deg": ["thigh_flexion_recalc_deg"],
    "thigh_extension_recalc_deg": ["thigh_extension_recalc_deg"],
    "knee_landing_event_recalc_deg": ["knee_landing_event_recalc_deg"],
    "shank_angle_median_deg": ["shank_angle_median_deg"],
    "os_event_abs_mean_mm": ["os_event_abs_mean_mm"],
    "os_event_abs_median_mm": ["os_event_abs_median_mm"],
    "os_event_abs_iqr_mm": ["os_event_abs_iqr_mm"],
    "os_event_signed_mean_mm": ["os_event_signed_mean_mm"],
    "os_left_abs_median_mm": ["os_left_abs_median_mm"],
    "os_right_abs_median_mm": ["os_right_abs_median_mm"],
    "os_lr_abs_diff_mm": ["os_lr_abs_diff_mm"],
    "os_clip_avg_mm": ["os_clip_avg_mm", "overstride_avg_mm_est"],
    "os_clip_trimmed_mean_mm": ["os_clip_trimmed_mean_mm", "overstride_trimmed_mean_mm_est"],
    "os_clip_selected_mm": ["os_clip_selected_mm", "overstride_selected_mm_est"],
    "foot_angle_abs_median_deg": ["foot_angle_abs_median_deg"],
    "foot_angle_iqr_deg": ["foot_angle_iqr_deg"],
    "shank_angle_abs_median_deg": ["shank_angle_abs_median_deg"],
    "knee_landing_event_median_deg": ["knee_landing_event_median_deg"],
    "contact_time_event_median_ms": ["contact_time_event_median_ms"],
    "height_cm": ["height_cm"],
    "running_speed_kmh": ["running_speed_kmh"],
    "side_pose_rate": ["side_pose_rate", "pose_detection_rate"],
    "side_actual_fps": ["side_actual_fps", "actual_video_fps"],
}


def _same_number(a, b, atol):
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=atol))


def _compare_reference(audit: dict, path: str, patient_id: str, atol: float):
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        raise ValueError("reference CSV requires patient_id column")
    row = df[df["patient_id"].astype(str) == str(patient_id)]
    if len(row) != 1:
        raise ValueError(f"reference patient row count={len(row)}")
    row = row.iloc[0]
    compared = []
    failed = []
    for key, value in audit.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        candidates = REFERENCE_ALIASES.get(key, [key])
        ref_key = next((name for name in candidates if name in row.index), None)
        if ref_key is None:
            continue
        ref = row[ref_key]
        if pd.isna(ref) and value is None:
            ok = True
        elif value is None or pd.isna(ref):
            ok = False
        else:
            ok = bool(np.isclose(float(value), float(ref), rtol=0.0, atol=atol))
        compared.append({"adapter_field": key, "reference_field": ref_key})
        if not ok:
            failed.append({
                "adapter_field": key,
                "reference_field": ref_key,
                "adapter": value,
                "reference": None if pd.isna(ref) else float(ref),
            })
    return {"compared_fields": compared, "failed": failed, "pass": len(failed) == 0}


def _sequence_array(event):
    return np.asarray(
        [[np.nan if value is None else float(value) for value in row] for row in event["sequence"]],
        dtype=np.float32,
    )


def _compare_sequence_reference(request: dict, npz_path: str, metadata_path: str, atol: float):
    data = np.load(npz_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    metadata = pd.read_csv(metadata_path)
    metadata["patient_id"] = metadata["patient_id"].astype(str)
    patient_id = str(request["patient_meta"]["patient_id"])
    patient_meta = metadata[metadata["patient_id"] == patient_id].copy()
    if patient_meta.empty:
        raise ValueError(f"patient_id={patient_id} not found in sequence reference")
    compared_events = 0
    max_abs_diff = 0.0
    nan_pattern_mismatch = 0
    rule_mismatch = 0
    count_mismatch = []
    for foot_input in request["strike_feet"]:
        foot = foot_input["foot"]
        ref = patient_meta[patient_meta["foot"].astype(str) == foot].sort_values("seq_index")
        events = foot_input["events"]
        if len(ref) != len(events):
            count_mismatch.append({"foot": foot, "adapter": len(events), "reference": len(ref)})
            continue
        for (_, ref_row), event in zip(ref.iterrows(), events):
            arr = _sequence_array(event)
            expected = X[int(ref_row["seq_index"])]
            if not np.array_equal(np.isnan(arr), np.isnan(expected)):
                nan_pattern_mismatch += 1
            mask = np.isfinite(arr) & np.isfinite(expected)
            if mask.any():
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(arr[mask] - expected[mask]))))
            expected_rule = None if pd.isna(ref_row.get("rule_class")) else str(ref_row.get("rule_class"))
            if event.get("rule_class") != expected_rule:
                rule_mismatch += 1
            compared_events += 1
    passed = (
        len(count_mismatch) == 0
        and nan_pattern_mismatch == 0
        and rule_mismatch == 0
        and max_abs_diff <= atol
    )
    return {
        "compared_events": compared_events,
        "max_absolute_difference": max_abs_diff,
        "nan_pattern_mismatch_events": nan_pattern_mismatch,
        "rule_mismatch_events": rule_mismatch,
        "count_mismatch": count_mismatch,
        "pass": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("debug_export")
    parser.add_argument("--reference-csv")
    parser.add_argument("--reference-sequence-npz")
    parser.add_argument("--reference-sequence-metadata")
    parser.add_argument("--api-url")
    parser.add_argument("--token")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--output", default="parity_result.json")
    args = parser.parse_args()

    if bool(args.reference_sequence_npz) != bool(args.reference_sequence_metadata):
        raise ValueError("--reference-sequence-npz and --reference-sequence-metadata must be supplied together")

    os.environ["RUNNINGAI_API_TOKEN"] = "parity-test-token"
    adapter_a = DebugExportAdapter.from_path(args.debug_export).build()
    adapter_b = DebugExportAdapter.from_path(args.debug_export).build()

    request_a = adapter_a.request.model_dump(mode="json")
    request_b = adapter_b.request.model_dump(mode="json")
    adapter_deterministic = request_a == request_b and adapter_a.audit == adapter_b.audit

    engine = get_engine()
    direct_overstride = engine.predict_overstride(request_a["overstride_features"])
    direct_strike = engine.predict_strike(request_a["strike_feet"])
    api_json = analyze(adapter_a.request).model_dump(mode="json")

    overstride_fields = ["prediction_mm", "prediction_std_mm", "model_pair_count", "status", "reason"]
    overstride_match = all(
        _same_number(api_json["overstride"].get(key), direct_overstride.get(key), args.atol)
        if key in {"prediction_mm", "prediction_std_mm"}
        else api_json["overstride"].get(key) == direct_overstride.get(key)
        for key in overstride_fields
    )

    strike_match = (
        api_json["strike_type"]["status"] == direct_strike["status"]
        and api_json["strike_type"]["patient_anchor_class"] == direct_strike["patient_anchor_class"]
        and _same_number(api_json["strike_type"]["patient_anchor_confidence"], direct_strike["patient_anchor_confidence"], args.atol)
        and _same_number(api_json["strike_type"]["rescue_p_heel"], direct_strike["rescue_p_heel"], args.atol)
        and api_json["strike_type"]["rescue_activated"] == direct_strike["rescue_activated"]
        and api_json["strike_type"]["final_class"] == direct_strike["final_class"]
        and _same_number(api_json["strike_type"]["final_confidence"], direct_strike["final_confidence"], args.atol)
        and api_json["strike_type"]["review_required"] == direct_strike["review_required"]
        and api_json["strike_type"]["feet"] == direct_strike["feet"]
    )

    client = TestClient(app)
    headers = {"Authorization": "Bearer parity-test-token"}
    upload = client.post(
        "/api/v1/adapter/debug-export",
        headers=headers,
        files={"file": (Path(args.debug_export).name, Path(args.debug_export).read_bytes(), "application/zip")},
    )
    if upload.status_code != 200:
        raise RuntimeError(upload.text)
    adapter_http_match = upload.json() == request_a

    external_http_match = None
    if args.api_url:
        token = args.token or os.environ.get("RUNNINGAI_API_TOKEN")
        if not token:
            raise ValueError("--token or RUNNINGAI_API_TOKEN is required with --api-url")
        response = requests.post(
            args.api_url.rstrip("/") + "/api/v1/analyze",
            headers={"Authorization": f"Bearer {token}"},
            json=request_a,
            timeout=180,
        )
        response.raise_for_status()
        external_http_match = response.json() == api_json

    reference = None
    if args.reference_csv:
        reference = _compare_reference(adapter_a.audit, args.reference_csv, request_a["patient_meta"]["patient_id"], args.atol)

    sequence_reference = None
    if args.reference_sequence_npz:
        sequence_reference = _compare_sequence_reference(
            request_a,
            args.reference_sequence_npz,
            args.reference_sequence_metadata,
            args.atol,
        )

    result = {
        "patient_id": request_a["patient_meta"]["patient_id"],
        "source_sha256": adapter_a.source_sha256,
        "adapter_deterministic": adapter_deterministic,
        "adapter_http_match": adapter_http_match,
        "overstride_direct_api_match": overstride_match,
        "strike_direct_api_match": strike_match,
        "external_http_match": external_http_match,
        "reference": reference,
        "sequence_reference": sequence_reference,
    }
    result["pass"] = all([
        adapter_deterministic,
        adapter_http_match,
        overstride_match,
        strike_match,
        True if external_http_match is None else external_http_match,
        True if reference is None else reference["pass"],
        True if sequence_reference is None else sequence_reference["pass"],
    ])
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
