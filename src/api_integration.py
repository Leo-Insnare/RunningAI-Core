from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .storage import EXPORTS_DIR, session_path, write_json

DEFAULT_API_URL = "http://127.0.0.1:8000"
SUPPORTED_MODES = {"embedded", "api"}


def resolve_inference_settings(secrets: Any | None = None) -> tuple[str, str, str]:
    mode = os.environ.get("RUNNINGAI_INFERENCE_MODE", "").strip().lower()
    api_url = os.environ.get("RUNNINGAI_API_URL", "").strip()
    token = os.environ.get("RUNNINGAI_API_TOKEN", "").strip()
    if secrets is not None:
        try:
            mode = mode or str(secrets.get("RUNNINGAI_INFERENCE_MODE", "")).strip().lower()
            api_url = api_url or str(secrets.get("RUNNINGAI_API_URL", "")).strip()
            token = token or str(secrets.get("RUNNINGAI_API_TOKEN", "")).strip()
        except Exception:
            pass
    mode = mode or "embedded"
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported inference mode: {mode}")
    return mode, api_url or DEFAULT_API_URL, token


def health(api_url: str, timeout: int = 5) -> dict:
    response = requests.get(api_url.rstrip("/") + "/health", timeout=timeout)
    response.raise_for_status()
    return response.json()


def model_info(api_url: str, token: str, timeout: int = 180) -> dict:
    response = requests.get(
        api_url.rstrip("/") + "/api/v1/model/info",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def analyze_debug_export(api_url: str, token: str, zip_path: Path, timeout: int = 180) -> dict:
    with zip_path.open("rb") as file_obj:
        response = requests.post(
            api_url.rstrip("/") + "/api/v1/analyze/debug-export",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (zip_path.name, file_obj, "application/zip")},
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()


def analyze_video_api(
    api_url: str,
    token: str,
    side_video_path: Path,
    patient_meta: dict,
    rear_video_path: Path | None = None,
    motionmetrix: dict | None = None,
    session_id: str | None = None,
    analysis_start_sec: float = 0.0,
    analysis_duration_sec: float = 60.0,
    timeout: int = 900,
) -> dict:
    files = {}
    handles = []
    try:
        side_handle = side_video_path.open("rb")
        handles.append(side_handle)
        files["side_video"] = (side_video_path.name, side_handle, "video/mp4")
        if rear_video_path is not None:
            rear_handle = rear_video_path.open("rb")
            handles.append(rear_handle)
            files["rear_video"] = (rear_video_path.name, rear_handle, "video/mp4")
        data = {
            "patient_meta": json.dumps(patient_meta, ensure_ascii=False),
            "motionmetrix": json.dumps(motionmetrix or {}, ensure_ascii=False),
            "analysis_start_sec": str(float(analysis_start_sec)),
            "analysis_duration_sec": str(float(analysis_duration_sec)),
        }
        if session_id:
            data["session_id"] = session_id
        response = requests.post(
            api_url.rstrip("/") + "/api/v1/analyze/video",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
            data=data,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    finally:
        for handle in handles:
            handle.close()


@lru_cache(maxsize=1)
def _embedded_engine():
    from api.app.model_engine import FrozenModelEngine

    return FrozenModelEngine()


def embedded_model_info() -> dict:
    from api.app.config import API_VERSION, MODEL_VERSION, load_status

    engine = _embedded_engine()
    status = load_status()
    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "stage": "final-independent-completed"
        if status.get("final_independent_test_completed")
        else "frozen-pre-final-independent",
        "supported_metrics": [
            "overstride",
            "strike_type",
            "cadence",
            "contact_time",
            "forward_lean",
            "max_thigh_flexion",
            "max_thigh_extension",
            "hip_rom",
            "knee_flexion_touchdown",
            "max_knee_flexion_swing",
            "knee_rom",
            "vertical_displacement",
            "pelvic_drop",
            "hip_hike_difference",
            "shank_angle_touchdown",
        ],
        "final_independent_test_completed": bool(
            status.get("final_independent_test_completed", False)
        ),
        "final_independent_targets_met": status.get("final_independent_targets_met"),
        "artifact_integrity": engine.status.artifact_integrity,
        "strike_aggregation": "bilateral patient-consensus",
        "strike_rescue_rule": "anchor=forefoot and rescue P(heel)>=0.65",
    }


def analyze_debug_export_embedded(zip_path: Path) -> dict:
    from api.app.config import MODEL_VERSION, REVIEW_CONFIDENCE_THRESHOLD
    from api.app.debug_export_adapter import DebugExportAdapter

    adapter = DebugExportAdapter.from_path(zip_path).build()
    req = adapter.request
    engine = _embedded_engine()

    features = dict(req.overstride_features)
    if req.patient_meta.height_cm is not None:
        features["height_cm"] = req.patient_meta.height_cm
    if req.patient_meta.running_speed_kmh is not None:
        features["running_speed_kmh"] = req.patient_meta.running_speed_kmh

    raw_overstride = engine.predict_overstride(features)
    quality_tier = req.quality.quality_tier
    overstride_review = raw_overstride["status"] != "completed" or quality_tier == "low"
    overstride = {
        "session_id": req.session_id,
        "patient_id": req.patient_meta.patient_id,
        "prediction_mm": raw_overstride["prediction_mm"],
        "prediction_std_mm": raw_overstride["prediction_std_mm"],
        "model_pair_count": raw_overstride["model_pair_count"],
        "quality_tier": quality_tier,
        "review_required": bool(overstride_review),
        "status": raw_overstride["status"],
        "reason": raw_overstride["reason"],
        "model_version": MODEL_VERSION,
    }

    raw_strike = engine.predict_strike([item.model_dump() for item in req.strike_feet])
    strike = {
        "session_id": req.session_id,
        "patient_id": req.patient_meta.patient_id,
        "status": raw_strike["status"],
        "reason": raw_strike["reason"],
        "feet": raw_strike["feet"],
        "patient_anchor_class": raw_strike["patient_anchor_class"],
        "patient_anchor_confidence": raw_strike["patient_anchor_confidence"],
        "final_class": raw_strike["final_class"],
        "final_confidence": raw_strike["final_confidence"],
        "review_required": raw_strike["review_required"],
        "rescue_p_heel": raw_strike["rescue_p_heel"],
        "rescue_activated": raw_strike["rescue_activated"],
        "aggregation": "patient_consensus",
        "model_version": MODEL_VERSION,
    }

    quality = req.quality.model_dump()
    reasons = list(quality.get("unmeasurable_reasons") or [])
    if overstride.get("status") == "unmeasurable" and overstride.get("reason"):
        reasons.append(f"overstride:{overstride['reason']}")
    if strike.get("status") == "unmeasurable" and strike.get("reason"):
        reasons.append(f"strike:{strike['reason']}")
    if overstride.get("status") == "unmeasurable" or strike.get("status") == "unmeasurable":
        quality["status"] = "unmeasurable"
    elif overstride.get("review_required") or strike.get("review_required") or quality_tier == "low":
        quality["status"] = "review_required"
    else:
        quality["status"] = "completed"
    quality["unmeasurable_reasons"] = list(dict.fromkeys(reasons))

    info = embedded_model_info()
    return {
        "session_id": req.session_id,
        "patient_id": req.patient_meta.patient_id,
        "overstride": overstride,
        "strike_type": strike,
        "posture_metrics": req.posture_metrics.model_dump(),
        "quality": quality,
        "model_info": info,
        "motionmetrix_comparison": {},
        "motionmetrix_reference": {},
        "source": {"input_type": "debug_export"},
    }


def analyze_session(zip_path: Path, mode: str, api_url: str = DEFAULT_API_URL, token: str = "") -> dict:
    if mode == "embedded":
        return analyze_debug_export_embedded(zip_path)
    if mode == "api":
        if not token:
            raise RuntimeError("RUNNINGAI_API_TOKEN is required in api mode.")
        return analyze_debug_export(api_url, token, zip_path)
    raise ValueError(f"Unsupported inference mode: {mode}")


def inference_status(mode: str, api_url: str = DEFAULT_API_URL, token: str = "") -> dict:
    if mode == "embedded":
        return embedded_model_info()
    if not token:
        raise RuntimeError("RUNNINGAI_API_TOKEN is required in api mode.")
    health(api_url)
    return model_info(api_url, token)


def analysis_summary_rows(result: dict) -> list[dict]:
    overstride = result.get("overstride") or {}
    strike = result.get("strike_type") or {}
    rows = [
        {
            "구분": "Overstride",
            "측면": "세션",
            "결과": overstride.get("prediction_mm"),
            "단위": "mm",
            "신뢰도": "",
            "검토 필요": bool(overstride.get("review_required", False)),
            "상태": overstride.get("status", ""),
        },
        {
            "구분": "Strike Type",
            "측면": "최종",
            "결과": strike.get("final_class"),
            "단위": "",
            "신뢰도": strike.get("final_confidence"),
            "검토 필요": bool(strike.get("review_required", False)),
            "상태": strike.get("status", ""),
        },
    ]
    for foot in strike.get("feet") or []:
        rows.append(
            {
                "구분": "Strike Type",
                "측면": foot.get("foot", ""),
                "결과": foot.get("prediction"),
                "단위": "",
                "신뢰도": foot.get("confidence"),
                "검토 필요": bool(foot.get("review_required", False)),
                "상태": strike.get("status", ""),
            }
        )
    return rows


def analysis_flat_row(result: dict) -> dict:
    overstride = result.get("overstride") or {}
    strike = result.get("strike_type") or {}
    quality = result.get("quality") or {}
    row = {
        "session_id": result.get("session_id"),
        "patient_id": result.get("patient_id"),
        "model_version": (result.get("model_info") or {}).get("model_version"),
        "overstride_prediction_mm": overstride.get("prediction_mm"),
        "overstride_prediction_std_mm": overstride.get("prediction_std_mm"),
        "overstride_status": overstride.get("status"),
        "overstride_review_required": overstride.get("review_required"),
        "strike_final_class": strike.get("final_class"),
        "strike_final_confidence": strike.get("final_confidence"),
        "strike_review_required": strike.get("review_required"),
        "strike_patient_anchor_class": strike.get("patient_anchor_class"),
        "strike_patient_anchor_confidence": strike.get("patient_anchor_confidence"),
        "strike_rescue_p_heel": strike.get("rescue_p_heel"),
        "strike_rescue_activated": strike.get("rescue_activated"),
        "quality_tier": quality.get("quality_tier"),
        "quality_status": quality.get("status"),
        "side_pose_rate": quality.get("side_pose_rate"),
        "rear_pose_rate": quality.get("rear_pose_rate"),
        "unmeasurable_reasons": ";".join(quality.get("unmeasurable_reasons") or []),
    }
    for foot in strike.get("feet") or []:
        name = str(foot.get("foot", "")).lower()
        if name not in {"left", "right"}:
            continue
        row[f"strike_{name}_prediction"] = foot.get("prediction")
        row[f"strike_{name}_confidence"] = foot.get("confidence")
        probabilities = foot.get("local_probabilities") or {}
        for cls in ["heel", "midfoot", "forefoot"]:
            row[f"strike_{name}_p_{cls}"] = probabilities.get(cls)
    posture = result.get("posture_metrics") or {}
    for key, value in posture.items():
        row[f"posture_{key}"] = value
    return row


def save_analysis_result(session_id: str, result: dict, export_dir: Path | None = None) -> list[Path]:
    out_dir = export_dir or EXPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{session_id}_api_analysis_result.json"
    csv_path = out_dir / f"{session_id}_api_analysis_result.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([analysis_flat_row(result)]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    if export_dir is None:
        write_json(session_path(session_id) / "api_analysis_result.json", result)
    return [json_path, csv_path]


def sync_analysis_sheet(result: dict, workbook_path: Path | None = None) -> Path | None:
    path = workbook_path or (EXPORTS_DIR / "final_comparison_summary.xlsx")
    if not path.exists():
        return None
    frame = pd.DataFrame([analysis_flat_row(result)])
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        frame.to_excel(writer, index=False, sheet_name="AI_Model_Result")
    return path


def load_saved_analysis(session_id: str) -> dict:
    path = session_path(session_id) / "api_analysis_result.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_result_export_parity(result: dict, csv_path: Path) -> dict:
    expected = analysis_flat_row(result)
    frame = pd.read_csv(csv_path)
    if len(frame) != 1:
        return {"pass": False, "mismatches": ["row_count"]}
    actual = frame.iloc[0].to_dict()
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if pd.isna(actual_value):
            actual_value = None
        if expected_value is None:
            if actual_value is not None:
                mismatches.append(key)
            continue
        # CSV blank cells are read back by pandas as NaN/None. Treat an
        # expected empty string as the same serialized value.
        if isinstance(expected_value, str) and expected_value == "" and actual_value is None:
            continue
        if isinstance(expected_value, bool):
            if bool(actual_value) != expected_value:
                mismatches.append(key)
            continue
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            try:
                if abs(float(actual_value) - float(expected_value)) > 1e-10:
                    mismatches.append(key)
            except Exception:
                mismatches.append(key)
            continue
        if str(actual_value) != str(expected_value):
            mismatches.append(key)
    return {"pass": not mismatches, "mismatches": mismatches}
