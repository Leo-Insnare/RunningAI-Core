from __future__ import annotations

import json
import math
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .debug_export_adapter import DebugExportAdapter
from .schemas import AnalysisSource, AnalyzeRequest, PatientMeta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_VIDEO_BYTES = 300 * 1024 * 1024
MAX_VIDEO_DURATION_SEC = 60.0
ALLOWED_VIDEO_SUFFIXES = {".mp4"}
SESSION_META_EXTRAS = {
    "side_progress_direction",
    "side_ground_y_px",
    "rear_ground_y_px",
    "contact_threshold_px",
    "side_scale_mm_per_px",
    "side_x_scale_mm_per_px",
    "side_y_scale_mm_per_px",
    "rear_scale_mm_per_px",
    "rear_x_scale_mm_per_px",
    "rear_y_scale_mm_per_px",
}


@dataclass
class VideoBuildResult:
    request: AnalyzeRequest
    source: AnalysisSource
    motionmetrix: dict[str, Any]


def parse_json_object(raw: str | None, field_name: str, required: bool = False) -> dict[str, Any]:
    if raw is None or not str(raw).strip():
        if required:
            raise ValueError(f"{field_name} is required.")
        return {}
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a valid JSON object.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return value


async def _save_upload(upload: Any, target: Path) -> int:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError("Only MP4 video files are supported.")
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with target.open("wb") as out:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_VIDEO_BYTES:
                raise ValueError("Video file exceeds the 300 MB limit.")
            out.write(chunk)
    if total <= 0:
        raise ValueError("Uploaded video is empty.")
    return total


def _probe_video(path: Path) -> dict[str, float | int | None]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from src.pose_preview import probe_video_info
    except Exception as exc:
        raise RuntimeError("OpenCV is required for raw video analysis.") from exc

    try:
        info = probe_video_info(path)
    except Exception as exc:
        raise ValueError("Uploaded MP4 could not be opened.") from exc

    fps = float(info.get("fps") or 0.0)
    frame_count = int(info.get("frame_count") or 0)
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    duration = info.get("duration_sec")
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Uploaded MP4 metadata is invalid.")
    if duration is not None and float(duration) > MAX_VIDEO_DURATION_SEC + 0.25:
        raise ValueError("Video duration exceeds the 60 second limit.")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": float(duration) if duration is not None else None,
        "rotation_clockwise_deg": int(info.get("rotation_clockwise_deg") or 0),
    }


def _effective_duration(probe: dict[str, Any], start_sec: float, requested_sec: float) -> float:
    if start_sec < 0:
        raise ValueError("analysis_start_sec must be >= 0.")
    if requested_sec < 0.5 or requested_sec > MAX_VIDEO_DURATION_SEC:
        raise ValueError("analysis_duration_sec must be between 0.5 and 60 seconds.")
    total = probe.get("duration_sec")
    if total is None:
        return float(requested_sec)
    total = float(total)
    if start_sec >= total:
        raise ValueError("analysis_start_sec is outside the video duration.")
    return max(0.5, min(float(requested_sec), total - start_sec, MAX_VIDEO_DURATION_SEC))


def _cleanup_processing_session(processing_session_id: str) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.storage import EXPORTS_DIR, session_path

    shutil.rmtree(session_path(processing_session_id), ignore_errors=True)
    for path in EXPORTS_DIR.glob(f"{processing_session_id}_*"):
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


async def build_analyze_request_from_videos(
    *,
    side_video: Any,
    rear_video: Any | None,
    session_id: str | None,
    patient_meta_raw: dict[str, Any],
    motionmetrix: dict[str, Any] | None,
    analysis_start_sec: float,
    analysis_duration_sec: float,
) -> VideoBuildResult:
    patient = PatientMeta.model_validate(patient_meta_raw)
    external_session_id = (session_id or "").strip() or f"API-{uuid.uuid4().hex[:12]}"
    processing_session_id = f"API_TMP_{uuid.uuid4().hex}"
    motionmetrix = dict(motionmetrix or {})

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.pose_preview import create_overlay_video, dependencies_status
    from src.session_exports import export_session_debug_files
    from src.storage import save_session, session_path, write_json

    deps = dependencies_status()
    if not deps.get("opencv") or not deps.get("mediapipe"):
        raise RuntimeError("Raw video analysis requires OpenCV and MediaPipe Pose.")

    base = session_path(processing_session_id)
    side_path = base / "videos" / "side_running.mp4"
    rear_path = base / "videos" / "rear_running.mp4"

    side_probe: dict[str, Any] = {}
    rear_probe: dict[str, Any] = {}
    effective_duration = float(analysis_duration_sec)
    try:
        await _save_upload(side_video, side_path)
        side_probe = _probe_video(side_path)
        effective_duration = _effective_duration(
            side_probe, float(analysis_start_sec), float(analysis_duration_sec)
        )

        if rear_video is not None:
            await _save_upload(rear_video, rear_path)
            rear_probe = _probe_video(rear_path)
            _effective_duration(
                rear_probe, float(analysis_start_sec), float(analysis_duration_sec)
            )

        session_meta: dict[str, Any] = {
            "session_id": external_session_id,
            "patient_id": patient.patient_id,
            "user_id": patient.patient_id,
            "height_cm": patient.height_cm,
            "weight_kg": patient.weight_kg,
            "running_speed_kmh": patient.running_speed_kmh,
            "side_video_fps": side_probe.get("fps"),
            "rear_video_fps": rear_probe.get("fps") if rear_probe else None,
            "side_measure_start_sec": float(analysis_start_sec),
            "side_measure_end_sec": float(analysis_start_sec) + effective_duration,
            "rear_measure_start_sec": float(analysis_start_sec),
            "rear_measure_end_sec": float(analysis_start_sec) + effective_duration,
        }
        for key in SESSION_META_EXTRAS:
            if key in patient_meta_raw and patient_meta_raw[key] not in (None, ""):
                session_meta[key] = patient_meta_raw[key]

        save_session(
            processing_session_id,
            session_meta,
            motionmetrix,
            {},
            {"missing_core_count": 0, "is_ready_for_export": True},
        )
        videos_json = {
            "side_running_video_path": str(side_path.relative_to(PROJECT_ROOT)),
            "side_running_original_filename": side_video.filename or "side_running.mp4",
        }
        if rear_video is not None:
            videos_json.update(
                {
                    "rear_running_video_path": str(rear_path.relative_to(PROJECT_ROOT)),
                    "rear_running_original_filename": rear_video.filename or "rear_running.mp4",
                }
            )
        write_json(base / "videos.json", videos_json)

        side_metric = {
            "metric_id": "foot_strike_type",
            "display_name_kr": "착지 형태",
            "display_name_en": "Foot Strike Type",
        }
        create_overlay_video(
            session_id=processing_session_id,
            video_path=side_path,
            metric=side_metric,
            start_sec=float(analysis_start_sec),
            duration_sec=effective_duration,
            output_fps=1.0,
            show_all=False,
            overlay_mode="summary",
        )

        if rear_video is not None:
            rear_effective_duration = _effective_duration(
                rear_probe, float(analysis_start_sec), float(analysis_duration_sec)
            )
            rear_metric = {
                "metric_id": "pelvic_drop",
                "display_name_kr": "골반 낙하",
                "display_name_en": "Pelvic Drop",
            }
            create_overlay_video(
                session_id=processing_session_id,
                video_path=rear_path,
                metric=rear_metric,
                start_sec=float(analysis_start_sec),
                duration_sec=rear_effective_duration,
                output_fps=1.0,
                show_all=False,
                overlay_mode="summary",
            )

        debug_paths = export_session_debug_files(processing_session_id)
        debug_zip = next(
            (p for p in reversed(debug_paths) if p.suffix.lower() == ".zip"), None
        )
        if debug_zip is None:
            raise RuntimeError("Raw video analysis did not produce a Debug Export package.")
        adapter = DebugExportAdapter.from_path(debug_zip).build()

        source = AnalysisSource(
            input_type="raw_video",
            side_video_filename=side_video.filename or "side_running.mp4",
            rear_video_filename=(rear_video.filename if rear_video is not None else None),
            analysis_start_sec=float(analysis_start_sec),
            analysis_duration_sec=float(effective_duration),
            side_video_fps=float(side_probe["fps"]),
            rear_video_fps=float(rear_probe["fps"]) if rear_probe else None,
        )
        return VideoBuildResult(
            request=adapter.request,
            source=source,
            motionmetrix=motionmetrix,
        )
    finally:
        _cleanup_processing_session(processing_session_id)


def _number(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _mean_values(*values: Any) -> float | None:
    nums = [_number(v) for v in values]
    nums = [v for v in nums if v is not None]
    return sum(nums) / len(nums) if nums else None


def _first_number(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in data:
            value = _number(data.get(key))
            if value is not None:
                return value
    return None


def build_motionmetrix_comparison(
    analysis: dict[str, Any], motionmetrix: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    mm = dict(motionmetrix or {})
    if not mm:
        return {}
    posture = analysis.get("posture_metrics") or {}
    overstride = analysis.get("overstride") or {}

    knee_swing_mm = _first_number(mm, "knee_flexion_swing_max_mean_deg")
    if knee_swing_mm is None:
        knee_swing_mm = _mean_values(
            mm.get("left_knee_flexion_swing_max_deg"),
            mm.get("right_knee_flexion_swing_max_deg"),
        )

    mapping = {
        "overstride": (
            _number(overstride.get("prediction_mm")),
            _first_number(mm, "overstride_mean_mm", "motionmetrix_overstride_mm"),
            "mm",
        ),
        "cadence": (
            _number(posture.get("cadence_spm")),
            _first_number(mm, "cadence_steps_per_min", "cadence_raw_value"),
            "steps/min",
        ),
        "contact_time": (
            _number(posture.get("contact_time_ms")),
            _first_number(mm, "contact_time_mean_ms"),
            "ms",
        ),
        "forward_lean": (
            _number(posture.get("forward_lean_deg")),
            _first_number(mm, "forward_lean_deg"),
            "deg",
        ),
        "hip_rom": (
            _number(posture.get("hip_rom_deg")),
            _first_number(mm, "hip_rom_mean_deg"),
            "deg",
        ),
        "knee_flexion_touchdown": (
            _number(posture.get("knee_flexion_touchdown_deg")),
            _first_number(mm, "knee_flexion_landing_mean_deg"),
            "deg",
        ),
        "max_knee_flexion_swing": (
            _number(posture.get("max_knee_flexion_swing_deg")),
            knee_swing_mm,
            "deg",
        ),
        "knee_rom": (
            _number(posture.get("knee_rom_deg")),
            _first_number(mm, "knee_rom_mean_deg"),
            "deg",
        ),
        "vertical_displacement": (
            _number(posture.get("vertical_displacement_mm")),
            _first_number(mm, "vertical_oscillation_mean_mm"),
            "mm",
        ),
        "pelvic_drop": (
            _number(posture.get("pelvic_drop_deg")),
            _first_number(mm, "pelvic_drop_mean_deg"),
            "deg",
        ),
        "shank_angle_touchdown": (
            _number(posture.get("shank_angle_touchdown_deg")),
            _first_number(mm, "shank_angle_mean_signed_deg", "shank_angle_mean_abs_deg"),
            "deg",
        ),
    }

    result: dict[str, dict[str, Any]] = {}
    for name, (analysis_value, mm_value, unit) in mapping.items():
        if mm_value is None:
            continue
        difference = (
            analysis_value - mm_value
            if analysis_value is not None and mm_value is not None
            else None
        )
        result[name] = {
            "analysis_value": analysis_value,
            "motionmetrix_value": mm_value,
            "difference": difference,
            "unit": unit,
        }
    return result
