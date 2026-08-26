import json
from pathlib import Path

import pandas as pd

from .definitions import all_metric_fields, all_session_fields, all_visual_fields
from .storage import EXPORTS_DIR, SESSIONS_DIR, read_json, video_paths_for_session
from .pose_preview import preview_summary_for_session
from .comparison import compute_auto_motionmetrix_values, export_final_comparison_summary


def _value(v):
    if v is None:
        return ""
    return v


VIDEO_EXPORT_FIELDS = [
    "side_static_video_path",
    "side_static_original_filename",
    "side_static_size_mb",
    "side_running_video_path",
    "side_running_original_filename",
    "side_running_size_mb",
    "rear_static_video_path",
    "rear_static_original_filename",
    "rear_static_size_mb",
    "rear_running_video_path",
    "rear_running_original_filename",
    "rear_running_size_mb",
]

VALIDATION_RANGES = {
    "age": (1, 120),
    "height_cm": (80, 230),
    "weight_kg": (20, 200),
    "running_speed_kmh": (3, 25),
    "side_video_fps": (15, 240),
    "rear_video_fps": (15, 240),
    "forward_lean_deg": (-30, 30),
    "running_economy_j_kg_m": (1.5, 6.0),
    "cadence_raw_value": (40, 260),
    "cadence_steps_per_min": (80, 260),
    "valid_measurement_time_sec": (1, 300),
}

KEYWORD_VALIDATION = [
    ("overstride", (-500, 800)),
    ("contact_time", (50, 1000)),
    ("shank_angle", (-60, 60)),
    ("knee_flexion", (0, 180)),
    ("knee_rom", (0, 180)),
    ("thigh_flexion", (-90, 140)),
    ("thigh_extension", (-90, 90)),
    ("hip_rom", (0, 180)),
    ("vertical_oscillation", (0, 300)),
    ("step_length", (0, 300)),
    ("stride_length", (0, 600)),
    ("flight_time", (0, 1000)),
    ("pelvic_drop", (-45, 45)),
    ("trunk_lateral_tilt", (-45, 45)),
    ("vertical_force", (0, 6)),
    ("lateral_force", (0, 3)),
    ("stride_rating", (0, 5)),
]


def validation_range(field_id):
    if field_id in VALIDATION_RANGES:
        return VALIDATION_RANGES[field_id]
    for keyword, value_range in KEYWORD_VALIDATION:
        if keyword in field_id:
            return value_range
    return None


def _float_or_none(value):
    if value in ["", None]:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _convert_fields(row, fields, unit_field, from_unit, factor):
    if row.get(unit_field) != from_unit:
        return
    for field in fields:
        value = _float_or_none(row.get(field))
        if value is not None:
            row[field] = value * factor


def _normalize_record(row):
    # v0.5.3: fill automatically computed customer-facing MotionMetrix values
    # such as Hip/Knee ROM averages and left-right differences.
    row.update(compute_auto_motionmetrix_values(row))

    raw = _float_or_none(row.get("cadence_raw_value"))
    raw_unit = row.get("cadence_raw_unit", "")
    if raw is not None and not row.get("cadence_steps_per_min"):
        if raw_unit == "strides/min":
            row["cadence_steps_per_min"] = raw * 2
        elif raw_unit == "steps/min":
            row["cadence_steps_per_min"] = raw

    _convert_fields(row, ["overstride_left_cm", "overstride_right_cm", "overstride_mean_cm", "overstride_asymmetry_cm"], "overstride_input_unit", "mm", 0.1)
    _convert_fields(row, ["left_contact_time_ms", "right_contact_time_ms", "contact_time_mean_ms", "contact_time_asymmetry_ms"], "contact_time_input_unit", "sec", 1000)
    _convert_fields(row, ["left_vertical_oscillation_cm", "right_vertical_oscillation_cm", "vertical_oscillation_mean_cm", "vertical_oscillation_asymmetry_cm"], "vertical_oscillation_input_unit", "mm", 0.1)
    _convert_fields(row, ["left_step_length_cm", "right_step_length_cm", "step_length_mean_cm", "step_length_asymmetry_cm", "left_stride_length_cm", "right_stride_length_cm", "stride_length_mean_cm"], "step_stride_length_input_unit", "m", 100)
    _convert_fields(row, ["left_after_flight_time_ms", "right_after_flight_time_ms", "flight_time_mean_ms", "flight_time_asymmetry_ms"], "flight_time_input_unit", "sec", 1000)
    return row


def build_session_record(session_id, metrics_defs, visual_defs):
    base = SESSIONS_DIR / session_id
    meta = read_json(base / "session_meta.json", {})
    values = read_json(base / "motionmetrix_values.json", {})
    visual = read_json(base / "visual_labels.json", {})
    videos = video_paths_for_session(session_id)
    row = {"session_id": session_id}
    for field in all_session_fields(metrics_defs):
        row[field["field_id"]] = _value(meta.get(field["field_id"], ""))
    for video_field in VIDEO_EXPORT_FIELDS:
        row[video_field] = _value(videos.get(video_field, meta.get(video_field, "")))
    for k, v in preview_summary_for_session(session_id).items():
        row[k] = _value(v)
    for field in all_metric_fields(metrics_defs):
        row[field["field_id"]] = _value(values.get(field["field_id"], ""))
    for field in all_visual_fields(visual_defs):
        row[field["field_id"]] = _value(visual.get(field["field_id"], ""))
    return _normalize_record(row)


def _target_session_dirs(session_ids=None):
    if session_ids:
        return [SESSIONS_DIR / sid for sid in session_ids if (SESSIONS_DIR / sid).is_dir()]
    return [p for p in SESSIONS_DIR.iterdir() if p.is_dir() and not p.name.startswith("SAMPLE_001")]


def export_wide(metrics_defs, visual_defs, session_ids=None):
    rows = [build_session_record(p.name, metrics_defs, visual_defs) for p in _target_session_dirs(session_ids)]
    path = EXPORTS_DIR / "training_dataset_wide.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_long(metrics_defs, visual_defs, session_ids=None):
    rows = []
    for session_dir in _target_session_dirs(session_ids):
        if not session_dir.is_dir():
            continue
        values = read_json(session_dir / "motionmetrix_values.json", {})
        videos = video_paths_for_session(session_dir.name)
        for video_field in VIDEO_EXPORT_FIELDS:
            rows.append({
                "session_id": session_dir.name,
                "metric_id": "video_file",
                "field_id": video_field,
                "display_name_kr": "영상 파일",
                "field_label_kr": video_field,
                "value": _value(videos.get(video_field, "")),
                "unit": "",
                "view": "session",
                "category": "video",
                "source_type": "uploaded_file",
                "model_role": "metadata",
                "required_level": "recommended",
                "required_points": "[]",
                "derived_points": "[]",
                "calculation_basis_kr": "세션과 연결된 측면/후면 영상 파일 경로",
                "ui_help_kr": "3차 모델링에서 영상과 라벨을 매칭하는 데 사용합니다.",
            })

        for preview_field, preview_value in preview_summary_for_session(session_dir.name).items():
            rows.append({
                "session_id": session_dir.name,
                "metric_id": "skeleton_preview",
                "field_id": preview_field,
                "display_name_kr": "Skeleton Preview 결과",
                "field_label_kr": preview_field,
                "value": _value(preview_value),
                "unit": "",
                "view": "preview",
                "category": "skeleton_preview",
                "source_type": "pose_preview",
                "model_role": "labeling_support",
                "required_level": "optional",
                "required_points": "[]",
                "derived_points": "[]",
                "calculation_basis_kr": "라벨링 보조용 skeleton overlay preview 결과",
                "ui_help_kr": "실시간/프레임 preview는 2차 라벨링 보조 기능이며 최종 학습 정답값은 MotionMetrix 입력값입니다.",
            })
        for metric in metrics_defs["metrics"]:
            for field in metric.get("fields", []):
                rows.append({
                    "session_id": session_dir.name,
                    "metric_id": metric["metric_id"],
                    "field_id": field["field_id"],
                    "display_name_kr": metric["display_name_kr"],
                    "field_label_kr": field["label_kr"],
                    "value": _value(values.get(field["field_id"], "")),
                    "unit": field.get("unit", ""),
                    "view": metric.get("view", ""),
                    "category": metric.get("category", ""),
                    "source_type": metric.get("source_type", ""),
                    "model_role": metric.get("model_role", ""),
                    "required_level": metric.get("required_level", ""),
                    "required_points": json.dumps(metric.get("required_points", []), ensure_ascii=False),
                    "derived_points": json.dumps(metric.get("derived_points", []), ensure_ascii=False),
                    "calculation_basis_kr": metric.get("calculation_basis_kr", ""),
                    "ui_help_kr": metric.get("ui_help_kr", ""),
                })
        visual = read_json(session_dir / "visual_labels.json", {})
        for field in all_visual_fields(visual_defs):
            rows.append({
                "session_id": session_dir.name,
                "metric_id": "visual_label",
                "field_id": field["field_id"],
                "display_name_kr": "육안 자세 평가",
                "field_label_kr": field["label_kr"],
                "value": _value(visual.get(field["field_id"], "")),
                "unit": "",
                "view": "visual",
                "category": "visual_label",
                "source_type": "manual_label",
                "model_role": "manual_label",
                "required_level": "manual_core" if field.get("required") else "optional",
                "required_points": "[]",
                "derived_points": "[]",
                "calculation_basis_kr": "원장님 또는 고객이 영상을 보고 직접 입력한 육안 자세 평가",
                "ui_help_kr": "MotionMetrix 정답값과 분리된 보조 라벨입니다.",
            })
    path = EXPORTS_DIR / "training_dataset_long.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_metric_dictionary(metrics_defs):
    rows = []
    for metric in metrics_defs["metrics"]:
        for field in metric.get("fields", []):
            rows.append({
                "metric_id": metric["metric_id"],
                "field_id": field["field_id"],
                "display_name_kr": metric["display_name_kr"],
                "display_name_en": metric["display_name_en"],
                "field_label_kr": field["label_kr"],
                "category": metric["category"],
                "view": metric["view"],
                "source_type": metric["source_type"],
                "model_role": metric["model_role"],
                "unit": field.get("unit", ""),
                "required_level": metric["required_level"],
                "field_required": field.get("required", False),
                "required_points": json.dumps(metric.get("required_points", []), ensure_ascii=False),
                "derived_points": json.dumps(metric.get("derived_points", []), ensure_ascii=False),
                "calculation_basis_kr": metric.get("calculation_basis_kr", ""),
                "ui_help_kr": metric.get("ui_help_kr", ""),
            })
    path = EXPORTS_DIR / "metric_dictionary.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def missing_values_for_session(session_id, metrics_defs, visual_defs):
    base = SESSIONS_DIR / session_id
    meta = read_json(base / "session_meta.json", {})
    values = read_json(base / "motionmetrix_values.json", {})
    visual = read_json(base / "visual_labels.json", {})
    rows = []
    for field in all_session_fields(metrics_defs):
        if field.get("required") and not meta.get(field["field_id"]):
            rows.append({"session_id": session_id, "section": "session_meta", "field_id": field["field_id"], "label_kr": field["label_kr"], "required_level": "core"})
    for metric in metrics_defs["metrics"]:
        if metric.get("required_level") not in ["core"]:
            continue
        for field in metric.get("fields", []):
            if field.get("required") and not values.get(field["field_id"]):
                rows.append({"session_id": session_id, "section": metric["metric_id"], "field_id": field["field_id"], "label_kr": field["label_kr"], "required_level": metric["required_level"]})
    for field in all_visual_fields(visual_defs):
        if field.get("required") and not visual.get(field["field_id"]):
            rows.append({"session_id": session_id, "section": "visual_labels", "field_id": field["field_id"], "label_kr": field["label_kr"], "required_level": "manual_core"})
    return rows


def export_missing_report(metrics_defs, visual_defs):
    rows = []
    for session_dir in SESSIONS_DIR.iterdir():
        if session_dir.is_dir():
            rows.extend(missing_values_for_session(session_dir.name, metrics_defs, visual_defs))
    columns = ["session_id", "section", "field_id", "label_kr", "required_level"]
    path = EXPORTS_DIR / "missing_value_report.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _count_range_warnings(record, metrics_defs, visual_defs):
    count = 0
    details = []
    fields = all_session_fields(metrics_defs) + all_metric_fields(metrics_defs)
    label_by_id = {f["field_id"]: f.get("label_kr", f["field_id"]) for f in fields}
    for field_id, value in record.items():
        rng = validation_range(field_id)
        if not rng:
            continue
        numeric = _float_or_none(value)
        if numeric is None:
            continue
        low, high = rng
        if numeric < low or numeric > high:
            count += 1
            details.append(f"{label_by_id.get(field_id, field_id)}={numeric} (권장 {low}~{high})")
    return count, "; ".join(details)


def export_data_quality_report(metrics_defs, visual_defs):
    rows = []
    summary = []
    session_dirs = [p for p in SESSIONS_DIR.iterdir() if p.is_dir()]
    ready_count = 0
    for session_dir in session_dirs:
        record = build_session_record(session_dir.name, metrics_defs, visual_defs)
        missing = missing_values_for_session(session_dir.name, metrics_defs, visual_defs)
        video_missing = [f for f in ["side_running_video_path", "rear_running_video_path"] if not record.get(f)]
        preview_summary = preview_summary_for_session(session_dir.name)
        range_count, range_details = _count_range_warnings(record, metrics_defs, visual_defs)
        is_ready = len(missing) == 0
        if is_ready:
            ready_count += 1
        rows.append({
            "session_id": session_dir.name,
            "is_ready_for_modeling": is_ready,
            "missing_core_count": len(missing),
            "video_missing_count": len(video_missing),
            "video_missing_fields": ", ".join(video_missing),
            "range_warning_count": range_count,
            "range_warning_details": range_details,
            "preview_count": preview_summary.get("preview_count", 0),
            "latest_preview_metric_id": preview_summary.get("latest_preview_metric_id", ""),
        })
    summary.append({
        "total_sessions": len(session_dirs),
        "modeling_ready_sessions": ready_count,
        "sessions_with_missing_core": sum(1 for r in rows if r["missing_core_count"] > 0),
        "sessions_with_video_missing": sum(1 for r in rows if r["video_missing_count"] > 0),
        "total_range_warnings": sum(r["range_warning_count"] for r in rows),
        "sessions_with_preview": sum(1 for r in rows if r.get("preview_count", 0) > 0),
    })
    detail_path = EXPORTS_DIR / "data_quality_report.csv"
    summary_path = EXPORTS_DIR / "data_quality_summary.csv"
    pd.DataFrame(rows).to_csv(detail_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return detail_path, summary_path


def export_all(metrics_defs, visual_defs, session_id: str | None = None):
    paths = [
        export_wide(metrics_defs, visual_defs),
        export_long(metrics_defs, visual_defs),
        export_metric_dictionary(metrics_defs),
        export_missing_report(metrics_defs, visual_defs),
    ]
    paths.extend(export_data_quality_report(metrics_defs, visual_defs))
    paths.extend(export_final_comparison_summary([session_id] if session_id else None))
    return paths
