from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

SIDE_DIRECT_TARGETS = [
    "Forward Lean", "Overstride", "Shank Angle", "Knee Flexion", "Contact Time",
    "Cadence", "Vertical Oscillation", "Braking Force", "Running Economy", "Running Type",
]
REAR_SKELETON_FEATURES = [
    "Pelvic Drop", "Trunk Lateral Tilt", "Shoulder Tilt",
    "Knee Alignment Angle", "Step Width / Crossover"
]


def pt(points: dict[int, dict[str, float]], idx: int):
    if idx not in points:
        return None
    return (float(points[idx]["x"]), float(points[idx]["y"]))


def midpoint(points: dict[int, dict[str, float]], a: int, b: int):
    pa, pb = pt(points, a), pt(points, b)
    if not pa or not pb:
        return None
    return ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)

def pt_world(points: dict[int, dict[str, float]], idx: int):
    if idx not in points:
        return None
    p = points[idx]
    if not all(k in p for k in ("world_x", "world_y", "world_z")):
        return None
    return (float(p["world_x"]), float(p["world_y"]), float(p["world_z"]))


def midpoint_world(points: dict[int, dict[str, float]], a: int, b: int):
    pa, pb = pt_world(points, a), pt_world(points, b)
    if not pa or not pb:
        return None
    return ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, (pa[2] + pb[2]) / 2.0)


def angle_between3d(a, b, c) -> float | None:
    if not a or not b or not c:
        return None
    v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]], dtype=float)
    v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]], dtype=float)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return None
    return float(math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))))


def angle_to_vertical3d(p_low, p_high, progress_sign: float = 1.0) -> float | None:
    """Signed small angle to the vertical axis using MediaPipe world x/y.

    v0.5.5: normalize the line angle to the closest small-angle representation.
    This prevents thigh/shank values such as +/-170 deg from producing impossible
    ROM values after max-min aggregation. MediaPipe world landmarks are still an
    estimated 3D source, not MotionMetrix-grade calibrated depth measurements.
    """
    if not p_low or not p_high:
        return None
    dx = p_high[0] - p_low[0]
    dy = p_low[1] - p_high[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return _small_angle_deg(float(progress_sign * math.degrees(math.atan2(dx, dy))))


def world_dx_m(p_from, p_to, progress_sign: float = 1.0) -> float | None:
    if not p_from or not p_to:
        return None
    return float(progress_sign * (p_to[0] - p_from[0]))


def angle_between(a, b, c) -> float | None:
    if not a or not b or not c:
        return None
    v1 = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    v2 = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return None
    return float(math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))))


def angle_to_vertical(p_low, p_high) -> float | None:
    if not p_low or not p_high:
        return None
    dx = p_high[0] - p_low[0]
    dy = p_low[1] - p_high[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return _small_angle_deg(float(math.degrees(math.atan2(dx, dy))))


def _small_angle_deg(angle: float | None) -> float | None:
    if angle is None:
        return None
    # Normalize line angle to the closest representation around 0 degrees.
    # This avoids rear pelvis angles such as 176 deg when the line is actually -4 deg from horizontal.
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle


def angle_to_horizontal(p1, p2) -> float | None:
    if not p1 or not p2:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return _small_angle_deg(float(math.degrees(math.atan2(-dy, dx))))


def line_offset_px(p, a, b) -> float | None:
    """Signed perpendicular distance from p to line a-b in pixels."""
    if not p or not a or not b:
        return None
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = bx - ax, by - ay
    denom = math.hypot(vx, vy)
    if denom == 0:
        return None
    return float(((px - ax) * vy - (py - ay) * vx) / denom)


def rear_knee_alignment_angle_deg(hip, knee, ankle) -> float | None:
    """Frontal-plane knee alignment angle proxy from rear image landmarks.

    MotionMetrix reports Knee Alignment @ mid-stance as a frontal-plane angle.
    With a single rear RGB video we cannot reproduce calibrated 3D valgus/varus,
    but we can expose a transparent 2D proxy: deviation of the hip-knee-ankle
    included angle from a straight line, signed by knee offset from the hip-ankle
    line. Positive/negative direction is for QA only; the absolute magnitude is
    generally more reliable for comparison.
    """
    included = angle_between(hip, knee, ankle)
    if included is None:
        return None
    offset = line_offset_px(knee, hip, ankle)
    sign = 1.0 if (offset or 0.0) >= 0 else -1.0
    return float(sign * max(0.0, 180.0 - included))


def _round(v: Any, ndigits: int = 3):
    if v is None:
        return ""
    try:
        if isinstance(v, (float, int, np.floating, np.integer)) and math.isfinite(float(v)):
            return round(float(v), ndigits)
    except Exception:
        return v
    return v


def _progress_sign(progress_direction: str | None) -> float:
    return -1.0 if str(progress_direction or "").lower() in {"right_to_left", "rtl", "right-left"} else 1.0


def compute_frame_metrics(points: dict[int, dict[str, float]], frame_index: int, timestamp_sec: float, source_fps: float, view_type: str, progress_direction: str = "left_to_right") -> dict[str, Any]:
    view_type = view_type or "unknown"
    psign = _progress_sign(progress_direction)
    shoulder = midpoint(points, 11, 12)
    pelvis = midpoint(points, 23, 24)
    shoulder_w = midpoint_world(points, 11, 12)
    pelvis_w = midpoint_world(points, 23, 24)
    world_available = bool(pelvis_w and shoulder_w)

    left_foot_low_y = max([points[i]["y"] for i in [29, 31, 27] if i in points], default=None)
    right_foot_low_y = max([points[i]["y"] for i in [30, 32, 28] if i in points], default=None)
    left_ankle, right_ankle = pt(points, 27), pt(points, 28)
    left_heel, right_heel = pt(points, 29), pt(points, 30)
    left_toe, right_toe = pt(points, 31), pt(points, 32)
    left_ankle_w, right_ankle_w = pt_world(points, 27), pt_world(points, 28)

    row: dict[str, Any] = {
        "frame_index": int(frame_index),
        "timestamp_sec": round(float(timestamp_sec), 3),
        "source_fps": round(float(source_fps or 0), 3),
        "view_type": view_type,
        "progress_direction": progress_direction or "left_to_right",
        "pose_detected": bool(points),
        "detected_points_count": len([k for k, v in points.items() if v.get("visibility", 0) >= 0.5]),
        "mediapipe_world_available": bool(world_available),
        "angle_calculation_source": "mediapipe_image_primary_world_audit" if world_available else "mediapipe_image",
        "distance_calculation_source": "mediapipe_world_estimate" if world_available else "mediapipe_image_px",
        "shoulder_center_x": _round(shoulder[0] if shoulder else None),
        "shoulder_center_y": _round(shoulder[1] if shoulder else None),
        "pelvis_center_x": _round(pelvis[0] if pelvis else None),
        "pelvis_center_y": _round(pelvis[1] if pelvis else None),
        "shoulder_center_world_x_m": _round(shoulder_w[0] if shoulder_w else None),
        "shoulder_center_world_y_m": _round(shoulder_w[1] if shoulder_w else None),
        "shoulder_center_world_z_m": _round(shoulder_w[2] if shoulder_w else None),
        "pelvis_center_world_x_m": _round(pelvis_w[0] if pelvis_w else None),
        "pelvis_center_world_y_m": _round(pelvis_w[1] if pelvis_w else None),
        "pelvis_center_world_z_m": _round(pelvis_w[2] if pelvis_w else None),
        "left_ankle_x": _round(left_ankle[0] if left_ankle else None),
        "left_ankle_y": _round(left_ankle[1] if left_ankle else None),
        "right_ankle_x": _round(right_ankle[0] if right_ankle else None),
        "right_ankle_y": _round(right_ankle[1] if right_ankle else None),
        "left_heel_x": _round(left_heel[0] if left_heel else None),
        "left_heel_y": _round(left_heel[1] if left_heel else None),
        "left_toe_x": _round(left_toe[0] if left_toe else None),
        "left_toe_y": _round(left_toe[1] if left_toe else None),
        "right_heel_x": _round(right_heel[0] if right_heel else None),
        "right_heel_y": _round(right_heel[1] if right_heel else None),
        "right_toe_x": _round(right_toe[0] if right_toe else None),
        "right_toe_y": _round(right_toe[1] if right_toe else None),
        "left_foot_low_y": _round(left_foot_low_y),
        "right_foot_low_y": _round(right_foot_low_y),
    }

    # Store selected world coordinates for downstream QA/modeling.
    for idx in [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
        wpt = pt_world(points, idx)
        if wpt:
            row[f"p{idx}_world_x_m"] = _round(wpt[0])
            row[f"p{idx}_world_y_m"] = _round(wpt[1])
            row[f"p{idx}_world_z_m"] = _round(wpt[2])

    # Image-based values are retained for audit/debug. Main angle fields prefer MediaPipe world landmarks when available.
    forward_img = angle_to_vertical(pelvis, shoulder)
    forward_world = angle_to_vertical3d(pelvis_w, shoulder_w, progress_sign=psign)
    lk_img = angle_between(pt(points, 23), pt(points, 25), pt(points, 27))
    rk_img = angle_between(pt(points, 24), pt(points, 26), pt(points, 28))
    lk_world = angle_between3d(pt_world(points, 23), pt_world(points, 25), pt_world(points, 27))
    rk_world = angle_between3d(pt_world(points, 24), pt_world(points, 26), pt_world(points, 28))
    ls_img = angle_to_vertical(pt(points, 27), pt(points, 25))
    rs_img = angle_to_vertical(pt(points, 28), pt(points, 26))
    ls_world = angle_to_vertical3d(pt_world(points, 27), pt_world(points, 25), progress_sign=psign)
    rs_world = angle_to_vertical3d(pt_world(points, 28), pt_world(points, 26), progress_sign=psign)
    lt_img = angle_to_vertical(pt(points, 23), pt(points, 25))
    rt_img = angle_to_vertical(pt(points, 24), pt(points, 26))
    lt_world = angle_to_vertical3d(pt_world(points, 23), pt_world(points, 25), progress_sign=psign)
    rt_world = angle_to_vertical3d(pt_world(points, 24), pt_world(points, 26), progress_sign=psign)

    row.update({
        "forward_lean_image_deg": _round(forward_img),
        "forward_lean_world_deg": _round(forward_world),
        # v0.5.7: keep world values for audit, but use side image-plane angles as
        # primary values for MotionMetrix-style sagittal-plane comparison.
        "forward_lean_deg": _round(psign * forward_img if forward_img is not None else forward_world),
        "left_knee_angle_image_deg": _round(lk_img),
        "right_knee_angle_image_deg": _round(rk_img),
        "left_knee_angle_world_deg": _round(lk_world),
        "right_knee_angle_world_deg": _round(rk_world),
        "left_knee_angle_deg": _round(lk_img if lk_img is not None else lk_world),
        "right_knee_angle_deg": _round(rk_img if rk_img is not None else rk_world),
        "left_knee_flexion_deg": _round(_knee_flexion_from_included(lk_img if lk_img is not None else lk_world)),
        "right_knee_flexion_deg": _round(_knee_flexion_from_included(rk_img if rk_img is not None else rk_world)),
        "left_shank_angle_image_deg": _round(_small_angle_deg(psign * ls_img) if ls_img is not None else None),
        "right_shank_angle_image_deg": _round(_small_angle_deg(psign * rs_img) if rs_img is not None else None),
        "left_shank_angle_world_deg": _round(ls_world),
        "right_shank_angle_world_deg": _round(rs_world),
        "left_shank_angle_deg": _round(_small_angle_deg(psign * ls_img) if ls_img is not None else ls_world),
        "right_shank_angle_deg": _round(_small_angle_deg(psign * rs_img) if rs_img is not None else rs_world),
        "left_thigh_angle_image_deg": _round(_small_angle_deg(psign * lt_img) if lt_img is not None else None),
        "right_thigh_angle_image_deg": _round(_small_angle_deg(psign * rt_img) if rt_img is not None else None),
        "left_thigh_angle_world_deg": _round(lt_world),
        "right_thigh_angle_world_deg": _round(rt_world),
        "left_thigh_angle_deg": _round(_small_angle_deg(psign * lt_img) if lt_img is not None else lt_world),
        "right_thigh_angle_deg": _round(_small_angle_deg(psign * rt_img) if rt_img is not None else rt_world),
        "left_foot_angle_deg": _round(angle_to_horizontal(left_heel, left_toe)),
        "right_foot_angle_deg": _round(angle_to_horizontal(right_heel, right_toe)),
    })

    # Side-view landing candidates and overstride proxy/estimate.
    if pelvis and left_ankle and right_ankle:
        active = "left" if (left_foot_low_y or -1) >= (right_foot_low_y or -1) else "right"
        ankle = left_ankle if active == "left" else right_ankle
        row["landing_foot_candidate"] = active
        row["landing_ankle_x"] = _round(ankle[0])
        row["landing_ankle_y"] = _round(ankle[1])
        row["pelvis_to_landing_ankle_dx_px"] = _round(psign * (ankle[0] - pelvis[0]))
        ankle_w = left_ankle_w if active == "left" else right_ankle_w
        dx_m = world_dx_m(pelvis_w, ankle_w, progress_sign=psign)
        row["pelvis_to_landing_ankle_dx_world_m"] = _round(dx_m)
        row["pelvis_to_landing_ankle_dx_mm_est"] = _round(dx_m * 1000.0 if dx_m is not None else None)
    else:
        row["landing_foot_candidate"] = ""
        row["landing_ankle_x"] = ""
        row["landing_ankle_y"] = ""
        row["pelvis_to_landing_ankle_dx_px"] = ""
        row["pelvis_to_landing_ankle_dx_world_m"] = ""
        row["pelvis_to_landing_ankle_dx_mm_est"] = ""

    # Rear-view features.
    # v0.5.12: expose every rear-view angle we can compute so the customer-facing
    # result table has no confusing blanks and can answer "후면도 나올 수 있는 각도".
    # These are Skeleton-derived 2D/estimated values, not MotionMetrix-grade depth values.
    row["rear_pelvic_tilt_deg"] = _round(angle_to_horizontal(pt(points, 23), pt(points, 24)))
    row["rear_shoulder_tilt_deg"] = _round(angle_to_horizontal(pt(points, 11), pt(points, 12)))
    row["rear_trunk_lateral_tilt_deg"] = _round(angle_to_vertical3d(pelvis_w, shoulder_w, progress_sign=1.0) if world_available else angle_to_vertical(pelvis, shoulder))
    left_knee_offset = line_offset_px(pt(points, 25), pt(points, 23), pt(points, 27))
    right_knee_offset = line_offset_px(pt(points, 26), pt(points, 24), pt(points, 28))
    row["left_knee_medial_offset_px"] = _round(left_knee_offset)
    row["right_knee_medial_offset_px"] = _round(right_knee_offset)
    row["left_knee_alignment_rear_deg"] = _round(rear_knee_alignment_angle_deg(pt(points, 23), pt(points, 25), pt(points, 27)))
    row["right_knee_alignment_rear_deg"] = _round(rear_knee_alignment_angle_deg(pt(points, 24), pt(points, 26), pt(points, 28)))
    if left_ankle and right_ankle:
        row["step_width_px"] = _round(abs(left_ankle[0] - right_ankle[0]))
        center_x = pelvis[0] if pelvis else (left_ankle[0] + right_ankle[0]) / 2.0
        row["crossover_flag"] = bool((left_ankle[0] - center_x) * (right_ankle[0] - center_x) > 0)
    else:
        row["step_width_px"] = ""
        row["crossover_flag"] = ""
    if left_ankle_w and right_ankle_w:
        # For rear view, x-axis separation is a MediaPipe world estimate, shown as mm for easier comparison/QA.
        row["step_width_world_m"] = _round(abs(left_ankle_w[0] - right_ankle_w[0]))
        row["step_width_mm_est"] = _round(abs(left_ankle_w[0] - right_ankle_w[0]) * 1000.0)
    else:
        row["step_width_world_m"] = ""
        row["step_width_mm_est"] = ""

    return row

def infer_contacts(frame_rows: list[dict[str, Any]], manual_ground_y_px: float | None = None, contact_threshold_px: float | None = None) -> list[dict[str, Any]]:
    if not frame_rows:
        return []
    left_y = [float(r["left_foot_low_y"]) for r in frame_rows if r.get("left_foot_low_y") not in ("", None)]
    right_y = [float(r["right_foot_low_y"]) for r in frame_rows if r.get("right_foot_low_y") not in ("", None)]
    all_y = left_y + right_y
    if not all_y:
        for r in frame_rows:
            r.update({"left_foot_contact": False, "right_foot_contact": False, "support_phase": "unknown", "active_support_foot": ""})
        return frame_rows
    ground_y = float(manual_ground_y_px) if manual_ground_y_px not in (None, "") else float(np.percentile(all_y, 95))
    threshold = float(contact_threshold_px) if contact_threshold_px not in (None, "") else max(8.0, abs(ground_y) * 0.015)
    prev_pelvis_x = None
    prev_t = None
    prev_pelvis_y = None
    for r in frame_rows:
        ly, ry = r.get("left_foot_low_y"), r.get("right_foot_low_y")
        left_contact = ly not in ("", None) and float(ly) >= ground_y - threshold
        right_contact = ry not in ("", None) and float(ry) >= ground_y - threshold
        if left_contact and not right_contact:
            phase, active = "left_support", "left"
        elif right_contact and not left_contact:
            phase, active = "right_support", "right"
        elif left_contact and right_contact:
            phase, active = "double_support", "both"
        else:
            phase, active = "flight", ""
        r.update({
            "ground_y_px": round(ground_y, 3),
            "contact_threshold_px": round(threshold, 3),
            "left_foot_contact": bool(left_contact),
            "right_foot_contact": bool(right_contact),
            "support_phase": phase,
            "active_support_foot": active,
        })
        px = r.get("pelvis_center_x")
        py = r.get("pelvis_center_y")
        ts = r.get("timestamp_sec")
        try:
            if prev_pelvis_x is not None and px not in ("", None) and prev_t is not None and float(ts) > prev_t:
                dt = float(ts) - prev_t
                r["pelvis_vx_px_s"] = round((float(px) - prev_pelvis_x) / dt, 3)
                if prev_pelvis_y is not None and py not in ("", None):
                    r["pelvis_vy_px_s"] = round((float(py) - prev_pelvis_y) / dt, 3)
                else:
                    r["pelvis_vy_px_s"] = ""
            else:
                r["pelvis_vx_px_s"] = ""
                r["pelvis_vy_px_s"] = ""
            if px not in ("", None):
                prev_pelvis_x = float(px)
            if py not in ("", None):
                prev_pelvis_y = float(py)
            prev_t = float(ts)
        except Exception:
            r["pelvis_vx_px_s"] = ""
            r["pelvis_vy_px_s"] = ""
    return frame_rows


def _mean(vals):
    nums = [float(v) for v in vals if v not in ("", None) and str(v).lower() != "nan"]
    return round(float(np.mean(nums)), 3) if nums else ""


def _max(vals):
    nums = [float(v) for v in vals if v not in ("", None) and str(v).lower() != "nan"]
    return round(float(np.max(nums)), 3) if nums else ""


def _min(vals):
    nums = [float(v) for v in vals if v not in ("", None) and str(v).lower() != "nan"]
    return round(float(np.min(nums)), 3) if nums else ""


def _nums(vals):
    out = []
    for v in vals:
        if v in ("", None) or str(v).lower() == "nan":
            continue
        try:
            fv = float(v)
            if math.isfinite(fv):
                out.append(fv)
        except Exception:
            continue
    return out


def _knee_flexion_from_included(angle):
    try:
        val = 180.0 - float(angle)
        # MotionMetrix reports knee flexion, not the raw hip-knee-ankle included angle.
        return round(max(0.0, min(180.0, val)), 3)
    except Exception:
        return ""


def _max_positive(vals):
    nums = _nums(vals)
    pos = [v for v in nums if v >= 0]
    return round(max(pos), 3) if pos else 0.0 if nums else ""


def _max_negative_magnitude(vals):
    nums = _nums(vals)
    neg = [abs(v) for v in nums if v < 0]
    return round(max(neg), 3) if neg else 0.0 if nums else ""


def _range(vals):
    nums = _nums(vals)
    return round(max(nums) - min(nums), 3) if nums else ""


def _median(vals):
    nums = _nums(vals)
    return round(float(np.median(nums)), 3) if nums else ""


def _percentile(vals, q: float):
    nums = _nums(vals)
    return round(float(np.percentile(nums, q)), 3) if nums else ""


def _trimmed_mean(vals, lower_q: float = 10, upper_q: float = 90, abs_value: bool = False):
    nums = _nums(vals)
    if not nums:
        return ""
    if abs_value:
        nums = [abs(v) for v in nums]
    lo = float(np.percentile(nums, lower_q))
    hi = float(np.percentile(nums, upper_q))
    kept = [v for v in nums if lo <= v <= hi]
    return round(float(np.mean(kept or nums)), 3)


def _median_abs(vals):
    nums = [abs(v) for v in _nums(vals)]
    return round(float(np.median(nums)), 3) if nums else ""


def _robust_median_abs(vals, lower_q: float = 10, upper_q: float = 90):
    nums = [abs(v) for v in _nums(vals)]
    if not nums:
        return ""
    if len(nums) >= 5:
        lo = float(np.percentile(nums, lower_q))
        hi = float(np.percentile(nums, upper_q))
        nums = [v for v in nums if lo <= v <= hi] or nums
    return round(float(np.median(nums)), 3)


def _event_side_representative(left_vals, right_vals, prefer_smaller_abs: bool = True):
    """Return a representative value and side when side-view occlusion makes both-feet averaging unstable.

    Side camera videos often see one foot more reliably. When left/right event
    values disagree strongly, averaging both sides can inflate MotionMetrix-style
    metrics such as overstride and shank angle. This helper chooses the side with
    the more plausible/stable representative while retaining the audit fields.
    """
    lm = _robust_median_abs(left_vals)
    rm = _robust_median_abs(right_vals)
    ln = len(_nums(left_vals))
    rn = len(_nums(right_vals))
    if lm == "" and rm == "":
        return "", "", "no_values"
    if lm == "":
        return rm, "right", "only_right_available"
    if rm == "":
        return lm, "left", "only_left_available"
    try:
        lf, rf = float(lm), float(rm)
    except Exception:
        return lm, "left", "fallback_left"
    # If one side is obviously inflated by occlusion, use the smaller magnitude side.
    if prefer_smaller_abs and (max(lf, rf) >= 1.8 * max(min(lf, rf), 1e-6) or abs(lf - rf) >= 50):
        return (lm, "left", "visible_side_smaller_magnitude") if lf <= rf else (rm, "right", "visible_side_smaller_magnitude")
    # Otherwise use the side with more events, or the median of both side representatives.
    if ln >= rn + 2:
        return lm, "left", "more_left_events"
    if rn >= ln + 2:
        return rm, "right", "more_right_events"
    return round(float(np.median([lf, rf])), 3), "bilateral", "bilateral_median"


def _signed_side_representative(left_vals, right_vals):
    """Representative for small signed angle metrics such as shank angle."""
    ln_vals = _nums(left_vals)
    rn_vals = _nums(right_vals)
    lm = round(float(np.median(ln_vals)), 3) if ln_vals else ""
    rm = round(float(np.median(rn_vals)), 3) if rn_vals else ""
    if lm == "" and rm == "":
        return "", "", "no_values"
    if lm == "":
        return rm, "right", "only_right_available"
    if rm == "":
        return lm, "left", "only_left_available"
    lf, rf = float(lm), float(rm)
    # If one visible side is much closer to vertical, prefer it; large L/R gaps are usually occlusion/view artifacts.
    if abs(lf - rf) >= 8:
        return (lm, "left", "visible_side_smaller_abs_angle") if abs(lf) <= abs(rf) else (rm, "right", "visible_side_smaller_abs_angle")
    return round(float(np.mean([lf, rf])), 3), "bilateral", "bilateral_mean"


def _bool_contact(v) -> bool:
    return str(v).lower() in {"true", "1", "yes"}


def _cycle_vertical_range_px(frame_rows: list[dict[str, Any]], events: list[dict[str, Any]], lower_q: float = 10, upper_q: float = 90):
    """Robust cycle-based pelvis vertical range in image pixels.

    MotionMetrix vertical displacement is closer to repeated COM oscillation
    amplitude than to whole-clip max-min. The old whole-clip max-min included
    MediaPipe jitter, drift, and framing shifts, which over-estimated mm values.
    """
    if not frame_rows:
        return ""
    frame_by_index = {}
    for i, r in enumerate(frame_rows):
        try:
            frame_by_index[int(r.get("frame_index"))] = i
        except Exception:
            pass
    event_frames = []
    for e in events or []:
        try:
            event_frames.append(int(float(e.get("initial_contact_frame"))))
        except Exception:
            pass
    event_frames = sorted(set(event_frames))
    ranges = []
    if len(event_frames) >= 2:
        for a, b in zip(event_frames, event_frames[1:]):
            ia, ib = frame_by_index.get(a), frame_by_index.get(b)
            if ia is None or ib is None or ib <= ia:
                continue
            vals = _nums([r.get("pelvis_center_y") for r in frame_rows[ia:ib + 1]])
            if len(vals) >= 4:
                ranges.append(float(np.percentile(vals, upper_q) - np.percentile(vals, lower_q)))
    if not ranges:
        vals = _nums([r.get("pelvis_center_y") for r in frame_rows])
        if len(vals) < 4:
            return ""
        return round(float(np.percentile(vals, upper_q) - np.percentile(vals, lower_q)), 3)
    return round(float(np.median(ranges)), 3)


def _body_height_px(row: dict[str, Any]) -> float | None:
    ys = []
    for key in [
        "shoulder_center_y", "pelvis_center_y",
        "left_ankle_y", "right_ankle_y", "left_heel_y", "right_heel_y", "left_toe_y", "right_toe_y",
    ]:
        v = row.get(key)
        if v not in ("", None):
            try:
                ys.append(float(v))
            except Exception:
                pass
    if len(ys) < 3:
        return None
    return max(ys) - min(ys)


def _event_foot_angle(row: dict[str, Any], foot: str):
    return row.get(f"{foot}_foot_angle_deg", "")


def _foot_strike_type(row: dict[str, Any], foot: str) -> str:
    heel_y = row.get(f"{foot}_heel_y")
    toe_y = row.get(f"{foot}_toe_y")
    if heel_y in ("", None) or toe_y in ("", None):
        return "unknown"
    diff = float(heel_y) - float(toe_y)
    if diff > 8:
        return "heel_candidate"
    if diff < -8:
        return "forefoot_candidate"
    return "midfoot_candidate"


def _window_mean_vx(rows: list[dict[str, Any]], center_idx: int, before: bool, n: int = 3):
    if before:
        subset = rows[max(0, center_idx - n):center_idx]
    else:
        subset = rows[center_idx + 1:min(len(rows), center_idx + 1 + n)]
    return _mean([r.get("pelvis_vx_px_s") for r in subset])


def _foot_series(frame_rows: list[dict[str, Any]], foot: str) -> list[float | None]:
    key = f"{foot}_foot_low_y"
    out: list[float | None] = []
    for r in frame_rows:
        try:
            out.append(float(r.get(key)))
        except Exception:
            out.append(None)
    return out


def _detect_contact_peaks(frame_rows: list[dict[str, Any]], foot: str, fps: float) -> list[int]:
    """Detect repeated touchdown candidates from image-y local maxima.

    This complements the ground-threshold contact state. It avoids the previous
    failure mode where one foot stayed classified as contact for the whole clip,
    causing cadence/contact time to collapse to one long event.
    """
    ys = _foot_series(frame_rows, foot)
    valid = [v for v in ys if v is not None]
    if len(valid) < 5:
        return []
    # v0.5.8: use a slightly broader near-ground threshold, but keep
    # cadence from over-counting by filtering alternating events later.
    near_ground = float(np.percentile(valid, 65))
    window = max(2, int(round(fps * 0.08)))
    min_gap = max(3, int(round(fps * 0.18)))
    peaks: list[int] = []
    for i, y in enumerate(ys):
        if y is None or y < near_ground:
            continue
        lo, hi = max(0, i - window), min(len(ys), i + window + 1)
        nbrs = [v for v in ys[lo:hi] if v is not None]
        if not nbrs or y < max(nbrs):
            continue
        if peaks and i - peaks[-1] < min_gap:
            if y > (ys[peaks[-1]] or -1):
                peaks[-1] = i
            continue
        peaks.append(i)
    return peaks


def _event_window_around_peak(frame_rows: list[dict[str, Any]], foot: str, peak_i: int, fps: float) -> tuple[int, int]:
    contact_key = f"{foot}_foot_contact"
    y_key = f"{foot}_foot_low_y"
    max_half = max(2, int(round(fps * 0.22)))
    start_i = peak_i
    while start_i > 0 and bool(frame_rows[start_i - 1].get(contact_key)) and peak_i - start_i < max_half:
        start_i -= 1
    end_i = peak_i
    while end_i < len(frame_rows) - 1 and bool(frame_rows[end_i + 1].get(contact_key)) and end_i - peak_i < max_half:
        end_i += 1

    # v0.5.8: if the binary contact flag is too brief, include the local
    # near-ground plateau around the peak. This prevents contact time from being
    # shortened to one frame on 24/30fps smartphone videos.
    # Low-FPS smartphone video quantizes contact time heavily. Use a slightly
    # wider minimum local window so contact duration does not collapse to 1-3 frames.
    min_contact_frames = max(2, int(round(fps * 0.20)))
    if end_i - start_i + 1 < min_contact_frames:
        try:
            ys = [float(r.get(y_key)) for r in frame_rows if r.get(y_key) not in ("", None)]
            peak_y = float(frame_rows[peak_i].get(y_key))
            if ys:
                local_range = max(1.0, float(np.percentile(ys, 95) - np.percentile(ys, 50)))
                plateau_tol = max(6.0, local_range * 0.18)
                start_i = peak_i
                while start_i > 0 and peak_i - start_i < max_half:
                    y_prev = frame_rows[start_i - 1].get(y_key)
                    if y_prev in ("", None) or float(y_prev) < peak_y - plateau_tol:
                        break
                    start_i -= 1
                end_i = peak_i
                while end_i < len(frame_rows) - 1 and end_i - peak_i < max_half:
                    y_next = frame_rows[end_i + 1].get(y_key)
                    if y_next in ("", None) or float(y_next) < peak_y - plateau_tol:
                        break
                    end_i += 1
        except Exception:
            pass

    if end_i - start_i + 1 < min_contact_frames:
        need = min_contact_frames - (end_i - start_i + 1)
        add_left = need // 2
        add_right = need - add_left
        start_i = max(0, start_i - add_left)
        end_i = min(len(frame_rows) - 1, end_i + add_right)
    # If contact stayed true for a long static segment, cap to a plausible local window.
    if end_i - start_i + 1 > max(3, int(round(fps * 0.45))):
        start_i = max(0, peak_i - int(round(fps * 0.10)))
        end_i = min(len(frame_rows) - 1, peak_i + int(round(fps * 0.25)))

    # v0.5.9: low-FPS smartphone video tends to cut toe-off one frame early.
    # Add one endpoint frame for running-event contact time, but keep the raw
    # start/end frames in the event CSV for audit. At 30fps this is ~33 ms, which
    # matches the observed MotionMetrix gap better than a 1-3 frame contact.
    if fps < 60 and end_i < len(frame_rows) - 1:
        # v0.5.10: 30fps videos quantize contact time heavily. 14/15번
        # 디버그에서 toe-off가 1~2 frames 짧게 끊기는 패턴이 확인되어
        # 저FPS에서는 plateau endpoint를 최대 2 frames까지 보강한다.
        extra = 2 if fps <= 35 else 1
        end_i = min(len(frame_rows) - 1, end_i + extra)
    return start_i, end_i


def _filter_alternating_events(events: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    """Remove obvious duplicate contacts while preserving realistic running cadence.

    Consecutive step interval in running is commonly ~0.25-0.45s, and the same
    foot should not touch down again for roughly twice that interval. The filter
    is intentionally conservative; it does not force values to MotionMetrix.
    """
    if not events:
        return []
    events = sorted(events, key=lambda e: float(e.get("initial_contact_time_sec", 0) or 0))
    min_any_gap = max(1.0 / max(fps, 1.0), 0.18)
    min_same_foot_gap = max(0.38, min(0.65, 0.45))
    kept: list[dict[str, Any]] = []
    last_by_foot: dict[str, float] = {}
    for ev in events:
        try:
            t = float(ev.get("initial_contact_time_sec", 0) or 0)
        except Exception:
            continue
        foot = str(ev.get("foot", ""))
        if kept:
            try:
                if t - float(kept[-1].get("initial_contact_time_sec", 0) or 0) < min_any_gap:
                    # Keep the event with longer contact window when two contacts
                    # are implausibly close.
                    if float(ev.get("contact_time_ms", 0) or 0) > float(kept[-1].get("contact_time_ms", 0) or 0):
                        kept[-1] = ev
                        last_by_foot[foot] = t
                    continue
            except Exception:
                pass
        if foot in last_by_foot and t - last_by_foot[foot] < min_same_foot_gap:
            continue
        kept.append(ev)
        if foot:
            last_by_foot[foot] = t
    return kept

def detect_gait_events(frame_rows: list[dict[str, Any]], view_type: str = "side") -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not frame_rows:
        return events
    try:
        fps = max(float(frame_rows[0].get("source_fps", 30) or 30), 1.0)
    except Exception:
        fps = 30.0

    # v0.5.7: primary event detection uses repeated local foot-y peaks.
    peak_events: list[dict[str, Any]] = []
    for foot in ["left", "right"]:
        for peak_i in _detect_contact_peaks(frame_rows, foot, fps):
            start_i, end_i = _event_window_around_peak(frame_rows, foot, peak_i, fps)
            peak_events.append(_build_event(frame_rows, start_i, end_i, foot, len(peak_events) + 1, view_type, event_method="foot_y_local_peak"))
    peak_events.sort(key=lambda e: e.get("initial_contact_time_sec", 0))
    raw_peak_count = len(peak_events)
    peak_events = _filter_alternating_events(peak_events, fps)
    # Use peak events if at least two are found. Otherwise keep the old transition
    # fallback, but mark it as low-confidence in _build_event.
    if len(peak_events) >= 2:
        for idx, event in enumerate(peak_events, start=1):
            event["event_id"] = idx
            event["event_quality"] = "medium_peak_based" if len(peak_events) >= 4 else "low_few_events"
            event["raw_peak_event_count_before_filter"] = raw_peak_count
            event["filtered_peak_event_count"] = len(peak_events)
        return peak_events

    for foot in ["left", "right"]:
        contact_key = f"{foot}_foot_contact"
        prev = bool(frame_rows[0].get(contact_key))
        start_i = 0 if prev else None
        for i in range(1, len(frame_rows)):
            curr = bool(frame_rows[i].get(contact_key))
            if curr and not prev:
                start_i = i
            if prev and not curr and start_i is not None:
                events.append(_build_event(frame_rows, start_i, i - 1, foot, len(events) + 1, view_type, event_method="contact_threshold_transition"))
                start_i = None
            prev = curr
        if start_i is not None and start_i < len(frame_rows) - 1:
            events.append(_build_event(frame_rows, start_i, len(frame_rows) - 1, foot, len(events) + 1, view_type, incomplete=True, event_method="contact_threshold_clip_boundary"))
    events.sort(key=lambda e: e.get("initial_contact_time_sec", 0))
    for idx, event in enumerate(events, start=1):
        event["event_id"] = idx
        event["event_quality"] = "low_transition_fallback"
    return events

def _build_event(rows: list[dict[str, Any]], start_i: int, end_i: int, foot: str, event_id: int, view_type: str, incomplete: bool = False, event_method: str = "contact_threshold_transition") -> dict[str, Any]:
    start = rows[start_i]
    end = rows[end_i]
    try:
        fps = max(float(start.get("source_fps", 0) or 0), 1.0)
    except Exception:
        fps = 30.0
    contact_frame_count = max(1, int(end_i - start_i + 1))
    contact_ms = max(0.0, contact_frame_count / fps * 1000.0)

    # v0.5.9: MotionMetrix touchdown values are measured at the earliest
    # initial contact. If the detected peak is one or two frames late, knee
    # flexion can be over-estimated sharply. Choose the most extended knee
    # frame in a small touchdown candidate window around contact start.
    td_lo = max(0, start_i - 2)
    td_hi = min(len(rows) - 1, start_i + 2)
    touchdown_i = start_i
    best_knee = None
    for j in range(td_lo, td_hi + 1):
        val = rows[j].get(f"{foot}_knee_flexion_deg")
        try:
            fv = float(val)
        except Exception:
            continue
        if best_knee is None or fv < best_knee:
            best_knee = fv
            touchdown_i = j
    touchdown = rows[touchdown_i]

    early_end_i = min(end_i, start_i + max(1, int(round((end_i - start_i + 1) * 0.30))))
    early_end = rows[early_end_i]
    before_vx = _window_mean_vx(rows, start_i, before=True)
    after_vx = _window_mean_vx(rows, start_i, before=False)
    decel = ""
    try:
        if before_vx not in ("", None) and after_vx not in ("", None):
            decel = round(float(after_vx) - float(before_vx), 3)
    except Exception:
        pass
    knee_included = touchdown.get(f"{foot}_knee_angle_deg", "")
    knee_flexion = touchdown.get(f"{foot}_knee_flexion_deg", "")
    shank = start.get(f"{foot}_shank_angle_deg", "")
    ankle_x = start.get(f"{foot}_ankle_x", "")
    ankle_y = start.get(f"{foot}_ankle_y", "")
    pelvis_x = start.get("pelvis_center_x", "")
    pelvis_y = start.get("pelvis_center_y", "")
    dx = ""
    dx_mm = start.get("pelvis_to_landing_ankle_dx_mm_est", "")
    try:
        if start.get("pelvis_to_landing_ankle_dx_px") not in ("", None):
            dx = start.get("pelvis_to_landing_ankle_dx_px")
        elif ankle_x not in ("", None) and pelvis_x not in ("", None):
            dx = round(float(ankle_x) - float(pelvis_x), 3)
    except Exception:
        pass
    return {
        "event_id": event_id,
        "view_type": view_type,
        "foot": foot,
        "event_type": "initial_contact_to_toe_off" if not incomplete else "initial_contact_to_clip_end",
        "initial_contact_frame": start.get("frame_index", ""),
        "initial_contact_time_sec": start.get("timestamp_sec", ""),
        "knee_touchdown_selected_frame": touchdown.get("frame_index", ""),
        "knee_touchdown_selected_time_sec": touchdown.get("timestamp_sec", ""),
        "contact_window_start_frame": start.get("frame_index", ""),
        "contact_window_start_time_sec": start.get("timestamp_sec", ""),
        "toe_off_frame": end.get("frame_index", ""),
        "toe_off_time_sec": end.get("timestamp_sec", ""),
        "contact_frame_count": contact_frame_count,
        "contact_time_ms": round(contact_ms, 1),
        "initial_support_start_sec": start.get("timestamp_sec", ""),
        "initial_support_end_sec": early_end.get("timestamp_sec", ""),
        "support_duration_ms": round(max(0.0, (float(early_end.get("timestamp_sec", 0)) - float(start.get("timestamp_sec", 0))) * 1000.0), 1),
        "pelvis_center_x_at_contact": pelvis_x,
        "pelvis_center_y_at_contact": pelvis_y,
        "landing_ankle_x": ankle_x,
        "landing_ankle_y": ankle_y,
        "pelvis_to_landing_ankle_dx_px": dx,
        "pelvis_to_landing_ankle_dx_mm_est": dx_mm,
        "overstride_at_contact_px": dx,
        "overstride_at_contact_mm_est": dx_mm,
        "knee_included_angle_at_contact_deg": knee_included,
        "knee_flexion_at_contact_deg": knee_flexion,
        "knee_angle_at_contact_deg": knee_flexion,
        "shank_angle_at_contact_deg": shank,
        "foot_angle_at_contact_deg": _event_foot_angle(start, foot),
        "foot_strike_type_estimate": _foot_strike_type(start, foot),
        "pelvis_vx_before_contact_px_s": before_vx,
        "pelvis_vx_after_contact_px_s": after_vx,
        "pelvis_forward_deceleration_px_s2_proxy": decel,
        "event_detection_method": event_method,
        "touchdown_selection_method": "min_knee_flexion_in_contact_window",
        "event_quality": "medium" if not incomplete else "low_clip_boundary",
        "confidence_contact": "medium" if not incomplete else "low_clip_boundary",
        "confidence_foot_strike": "medium",
        "confidence_event": "medium" if not incomplete else "low_clip_boundary",
        "source_fps": round(fps, 3),
        "timing_confidence": "low_fps" if fps < 60 else "medium" if fps < 100 else "high",
        "missing_reason": "" if not incomplete else "toe-off not observed before clip end",
    }


def build_second_summary(frame_rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for r in frame_rows:
        sec = int(float(r.get("timestamp_sec", 0)))
        groups.setdefault(sec, []).append(r)
    event_groups: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        try:
            sec = int(float(e.get("initial_contact_time_sec", 0)))
            event_groups.setdefault(sec, []).append(e)
        except Exception:
            pass
    out = []
    for sec in sorted(groups):
        rows = groups[sec]
        evs = event_groups.get(sec, [])
        valid = [r for r in rows if r.get("pose_detected")]
        duration = max(1e-6, (float(rows[-1].get("timestamp_sec", 0)) - float(rows[0].get("timestamp_sec", 0)) + 1.0 / max(float(rows[0].get("source_fps", 30) or 30), 1)))
        out.append({
            "second": sec,
            "frame_start": rows[0].get("frame_index", ""),
            "frame_end": rows[-1].get("frame_index", ""),
            "pose_detection_rate": round(len(valid) / max(len(rows), 1), 3),
            "mean_forward_lean_deg": _mean([r.get("forward_lean_deg") for r in rows]),
            "mean_left_knee_angle_deg": _mean([r.get("left_knee_angle_deg") for r in rows]),
            "mean_right_knee_angle_deg": _mean([r.get("right_knee_angle_deg") for r in rows]),
            "mean_left_shank_angle_deg": _mean([r.get("left_shank_angle_deg") for r in rows]),
            "mean_right_shank_angle_deg": _mean([r.get("right_shank_angle_deg") for r in rows]),
            "mean_pelvis_vx_px_s": _mean([r.get("pelvis_vx_px_s") for r in rows]),
            "mean_pelvis_vertical_position_px": _mean([r.get("pelvis_center_y") for r in rows]),
            "left_contact_frame_count": sum(1 for r in rows if r.get("left_foot_contact")),
            "right_contact_frame_count": sum(1 for r in rows if r.get("right_foot_contact")),
            "contact_event_count": len(evs),
            "estimated_cadence_spm": round(len(evs) / duration * 60.0, 3) if evs else "",
            "mean_contact_time_ms": _mean([e.get("contact_time_ms") for e in evs]),
            "mean_pelvis_to_ankle_dx_px": _mean([e.get("pelvis_to_landing_ankle_dx_px") for e in evs]),
        })
    return out


def build_clip_summary(frame_rows: list[dict[str, Any]], events: list[dict[str, Any]], motionmetrix_values: dict[str, Any] | None = None, session_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build clip-level Skeleton averages aligned to MotionMetrix screen definitions.

    v0.5.5 keeps the v0.5.4 UI but changes the aggregation logic:
    - Knee Flexion uses flexion angle (180 - included joint angle), not raw included angle.
    - Thigh/Hip ROM is flexion magnitude + extension magnitude, avoiding +/-180 wrapping.
    - Shank Angle, Overstride, and Knee Flexion @ touch-down use initial-contact events.
    - Vertical Displacement uses image pelvis/COM vertical range with height-based px→mm estimate.
    - Contact Time is displayed in seconds in final comparison, though ms is retained internally.
    """
    motionmetrix_values = motionmetrix_values or {}
    session_meta = session_meta or {}
    if not frame_rows:
        return {}
    duration = max(1e-6, float(frame_rows[-1].get("timestamp_sec", 0)) - float(frame_rows[0].get("timestamp_sec", 0)) + 1.0 / max(float(frame_rows[0].get("source_fps", 30) or 30), 1.0))
    # Keep all events for event CSV, but use plausible events for summary to reduce false contacts.
    valid_events = []
    for e in events:
        try:
            if float(e.get("contact_time_ms", 0) or 0) >= 30.0:
                valid_events.append(e)
        except Exception:
            pass
    events_for_summary = valid_events or events
    left_events = [e for e in events_for_summary if e.get("foot") == "left"]
    right_events = [e for e in events_for_summary if e.get("foot") == "right"]

    pelvis_y_nums = _nums([r.get("pelvis_center_y") for r in frame_rows])
    vertical_osc_px_clip_range = round(max(pelvis_y_nums) - min(pelvis_y_nums), 3) if pelvis_y_nums else ""
    # v0.5.9: use gait-cycle median vertical displacement instead of whole-clip
    # max-min range. This better approximates MotionMetrix COM oscillation and
    # reduces drift/jitter inflation from smartphone videos.
    vertical_osc_px_cycle_median = _cycle_vertical_range_px(frame_rows, events_for_summary)
    vertical_osc_px = vertical_osc_px_cycle_median if vertical_osc_px_cycle_median not in ("", None) else vertical_osc_px_clip_range
    body_heights = [_body_height_px(r) for r in frame_rows]
    body_height_px = _median([v for v in body_heights if v not in (None, "") and v > 0])
    height_cm = session_meta.get("height_cm") or motionmetrix_values.get("height_cm")
    view_for_scale = str(frame_rows[0].get("view_type", "") or "").lower()
    # v0.5.11: split distance scale by axis. Overstride is an x-axis distance,
    # while Vertical Displacement is a y-axis distance. A single height-based
    # mm/px scale can overstate overstride in smartphone side videos, so keep
    # side_x / side_y / rear_x / rear_y manual scales separate when available.
    legacy_manual_scale_key = "rear_scale_mm_per_px" if view_for_scale == "rear" else "side_scale_mm_per_px"
    x_scale_key = "rear_x_scale_mm_per_px" if view_for_scale == "rear" else "side_x_scale_mm_per_px"
    y_scale_key = "rear_y_scale_mm_per_px" if view_for_scale == "rear" else "side_y_scale_mm_per_px"

    def _manual_scale_from(*keys):
        for key in keys:
            val = session_meta.get(key) or motionmetrix_values.get(key)
            try:
                if val not in ("", None, 0):
                    return float(val), key
            except Exception:
                continue
        return None, ""

    x_px_to_mm, x_scale_used_key = _manual_scale_from(x_scale_key, legacy_manual_scale_key)
    y_px_to_mm, y_scale_used_key = _manual_scale_from(y_scale_key, legacy_manual_scale_key)
    px_to_mm = y_px_to_mm
    scale_source = f"manual_{y_scale_used_key}" if y_px_to_mm else "unavailable"
    scale_confidence = "high" if y_px_to_mm else "unavailable"
    if px_to_mm is None:
        try:
            height_mm = float(height_cm) * 10.0
            px_to_mm = height_mm / float(body_height_px) if body_height_px not in ("", None, 0) else None
        except Exception:
            px_to_mm = None
        scale_source = "height_based_y" if px_to_mm else "unavailable"
        # Smartphone video pixel-to-mm conversion is highly camera-position dependent.
        # Keep mm_est for comparison, but mark confidence unless a manual scale is provided.
        scale_confidence = "low" if px_to_mm else "unavailable"
    if x_px_to_mm is None:
        x_px_to_mm = px_to_mm
        x_scale_source = "height_based_x_from_y" if px_to_mm else "unavailable"
        x_scale_confidence = "low" if px_to_mm else "unavailable"
    else:
        x_scale_source = f"manual_{x_scale_used_key}"
        x_scale_confidence = "high"
    vertical_osc_mm = round(float(vertical_osc_px) * px_to_mm, 3) if vertical_osc_px not in ("", None) and px_to_mm else ""

    # v0.5.11: Hip ROM itself matched MotionMetrix well after v0.5.10,
    # but flexion/extension split was biased (e.g., flexion too small, extension
    # too large). Use p10~p90 ROM as total movement and a mid-range neutral for
    # the comparison split. This preserves ROM while avoiding running-median
    # neutral drift. Median-neutral values are retained for audit.
    left_thigh_vals = _nums([r.get("left_thigh_angle_deg") for r in frame_rows])
    right_thigh_vals = _nums([r.get("right_thigh_angle_deg") for r in frame_rows])

    def _robust_thigh(vals):
        if not vals:
            return "", "", "", "", "", "", ""
        p10 = float(np.percentile(vals, 10))
        p90 = float(np.percentile(vals, 90))
        med = float(np.median(vals))
        # Audit: previous median-neutral split.
        flex_median = max(0.0, p90 - med)
        ext_median_mag = max(0.0, med - p10)
        # Selected: mid-range neutral split, keeping total ROM stable.
        rom = max(0.0, p90 - p10)
        flex = rom / 2.0
        ext_mag = rom / 2.0
        ext_signed = -ext_mag
        return round(flex, 3), round(ext_signed, 3), round(ext_mag, 3), round(rom, 3), round(flex_median, 3), round(-ext_median_mag, 3), round(ext_median_mag, 3)

    left_thigh_flex, left_thigh_ext, left_thigh_ext_mag, left_hip_rom, left_thigh_flex_median, left_thigh_ext_median, left_thigh_ext_median_mag = _robust_thigh(left_thigh_vals)
    right_thigh_flex, right_thigh_ext, right_thigh_ext_mag, right_hip_rom, right_thigh_flex_median, right_thigh_ext_median, right_thigh_ext_median_mag = _robust_thigh(right_thigh_vals)

    left_knee_flex_vals = _nums([r.get("left_knee_flexion_deg") for r in frame_rows])
    right_knee_flex_vals = _nums([r.get("right_knee_flexion_deg") for r in frame_rows])
    left_stance_rows = [r for r in frame_rows if r.get("left_foot_contact")]
    right_stance_rows = [r for r in frame_rows if r.get("right_foot_contact")]
    left_swing_rows = [r for r in frame_rows if not r.get("left_foot_contact")]
    right_swing_rows = [r for r in frame_rows if not r.get("right_foot_contact")]
    # v0.5.10: use robust high-percentile instead of raw max so a single
    # MediaPipe jitter frame does not inflate MotionMetrix stance/swing max.
    left_knee_stance_max = _percentile([r.get("left_knee_flexion_deg") for r in left_stance_rows], 95)
    right_knee_stance_max = _percentile([r.get("right_knee_flexion_deg") for r in right_stance_rows], 95)
    left_knee_swing_max = _percentile([r.get("left_knee_flexion_deg") for r in left_swing_rows], 95)
    right_knee_swing_max = _percentile([r.get("right_knee_flexion_deg") for r in right_swing_rows], 95)
    left_knee_landing = _mean([e.get("knee_flexion_at_contact_deg", e.get("knee_angle_at_contact_deg")) for e in left_events])
    right_knee_landing = _mean([e.get("knee_flexion_at_contact_deg", e.get("knee_angle_at_contact_deg")) for e in right_events])
    left_knee_rom = round(abs(float(left_knee_swing_max) - float(left_knee_landing)), 3) if left_knee_swing_max not in ("", None) and left_knee_landing not in ("", None) else _range(left_knee_flex_vals)
    right_knee_rom = round(abs(float(right_knee_swing_max) - float(right_knee_landing)), 3) if right_knee_swing_max not in ("", None) and right_knee_landing not in ("", None) else _range(right_knee_flex_vals)

    contact_time_ms = _mean([e.get("contact_time_ms") for e in events_for_summary])
    contact_time_sec = round(float(contact_time_ms) / 1000.0, 3) if contact_time_ms not in ("", None) else ""
    cadence_count_spm = round(len(events_for_summary) / duration * 60.0, 3) if duration > 0 else ""
    event_times = _nums([e.get("initial_contact_time_sec") for e in events_for_summary])
    event_times = sorted(event_times)
    intervals = [b - a for a, b in zip(event_times, event_times[1:]) if b > a]
    cadence_interval_spm = round(60.0 / float(np.median(intervals)), 3) if len(intervals) >= 3 and float(np.median(intervals)) > 0 else ""
    edge_adjusted_event_count = len(events_for_summary)
    cadence_selection_method = "event_count_per_duration"
    if len(intervals) >= 3 and cadence_interval_spm not in ("", None):
        first_gap = event_times[0] - float(frame_rows[0].get("timestamp_sec", 0) or 0) if event_times else 0
        last_gap = float(frame_rows[-1].get("timestamp_sec", 0) or 0) - event_times[-1] if event_times else 0
        med_interval = float(np.median(intervals))
        # Short 5s clips often cut a step at the beginning/end. Add half-step
        # credits for large edge gaps and prefer interval cadence when it is plausible.
        edge_credit = 0.0
        if first_gap > 0.65 * med_interval:
            edge_credit += 0.5
        if last_gap > 0.65 * med_interval:
            edge_credit += 0.5
        edge_adjusted_event_count = round(len(events_for_summary) + edge_credit, 3)
        cadence_edge_spm = round(edge_adjusted_event_count / duration * 60.0, 3) if duration > 0 else ""
        # v0.5.11: median-step-interval can overestimate cadence in short 5s clips
        # when only a few intervals are available. Keep three candidates and use a
        # robust candidate median when they diverge; in 15번 this selects the
        # edge-adjusted value rather than the high interval-only value.
        cadence_candidates = [v for v in [cadence_count_spm, cadence_edge_spm, cadence_interval_spm] if v not in ("", None)]
        try:
            cand_nums = [float(v) for v in cadence_candidates]
        except Exception:
            cand_nums = []
        if len(cand_nums) >= 3 and (max(cand_nums) - min(cand_nums) > 20):
            cadence_spm = round(float(np.median(cand_nums)), 3)
            cadence_selection_method = "candidate_median_count_edge_interval"
        elif 120 <= float(cadence_interval_spm) <= 230 and (cadence_count_spm == "" or abs(float(cadence_interval_spm) - float(cadence_count_spm)) <= 25):
            cadence_spm = cadence_interval_spm
            cadence_selection_method = "median_step_interval"
        else:
            cadence_spm = cadence_edge_spm
            cadence_selection_method = "edge_adjusted_event_count"
    else:
        cadence_edge_spm = cadence_count_spm
        cadence_spm = cadence_count_spm
    event_quality_summary = "ok" if len(events_for_summary) >= 6 else "event_count_low" if events_for_summary else "event_not_detected"

    # v0.5.10: Shank Angle uses raw touchdown angle by default.
    # 14/15번에서 stance/static baseline correction over-corrected otherwise
    # well-aligned raw shank values. Apply baseline only when offset is small.
    stance_shanks = []
    for r in frame_rows:
        if r.get("left_foot_contact") and r.get("left_shank_angle_deg") not in ("", None):
            stance_shanks.append(r.get("left_shank_angle_deg"))
        if r.get("right_foot_contact") and r.get("right_shank_angle_deg") not in ("", None):
            stance_shanks.append(r.get("right_shank_angle_deg"))
    shank_baseline = _median(stance_shanks)
    baseline_num = None
    try:
        baseline_num = float(shank_baseline) if shank_baseline not in ("", None) else None
    except Exception:
        baseline_num = None
    shank_apply_baseline = bool(baseline_num is not None and abs(baseline_num) <= 8.0)
    def _corr_shank(v):
        try:
            if v in ("", None) or baseline_num is None:
                return v
            return round(float(v) - baseline_num, 3)
        except Exception:
            return v

    # v0.5.9: MotionMetrix overstride is a representative positive distance at
    # touch-down. Event values may include occlusion/outlier contacts, so use a
    # robust median/trimmed mean rather than a plain mean.
    # v0.5.11: compute Overstride from image x-axis px with a dedicated x-scale.
    # World/height-based mm can be too large in side videos. Keep raw px and
    # scale confidence so the customer can see whether manual x-scale is needed.
    overstride_px_vals = [e.get("pelvis_to_landing_ankle_dx_px") for e in events_for_summary]
    overstride_px_abs = _median_abs(overstride_px_vals)
    overstride_px_trimmed = _trimmed_mean(overstride_px_vals, abs_value=True)
    left_overstride_px_vals = [e.get("pelvis_to_landing_ankle_dx_px") for e in left_events]
    right_overstride_px_vals = [e.get("pelvis_to_landing_ankle_dx_px") for e in right_events]
    overstride_selected_px, overstride_selected_side, overstride_selection_reason = _event_side_representative(left_overstride_px_vals, right_overstride_px_vals, prefer_smaller_abs=True)

    def _scaled_abs(v, scale):
        try:
            if v in ("", None) or scale in ("", None):
                return ""
            return round(abs(float(v)) * float(scale), 3)
        except Exception:
            return ""

    if x_scale_confidence == "high":
        # Manual x-axis scale is the most transparent way to compare MotionMetrix mm.
        overstride_mm_abs = _scaled_abs(overstride_px_abs, x_px_to_mm)
        overstride_mm_trimmed = _scaled_abs(overstride_px_trimmed, x_px_to_mm)
        overstride_selected_mm = _scaled_abs(overstride_selected_px, x_px_to_mm)
        left_overstride_mm_vals = [_scaled_abs(v, x_px_to_mm) for v in left_overstride_px_vals]
        right_overstride_mm_vals = [_scaled_abs(v, x_px_to_mm) for v in right_overstride_px_vals]
        overstride_distance_source = "manual_x_px_scale"
    else:
        # Without manual x-scale, keep MediaPipe world dx estimate rather than
        # height-based image x scaling. It was closer in 14/15 debug data, but
        # remains low-confidence because it is not a calibrated depth measure.
        overstride_mm_vals = [e.get("pelvis_to_landing_ankle_dx_mm_est") for e in events_for_summary]
        left_overstride_mm_vals = [e.get("pelvis_to_landing_ankle_dx_mm_est") for e in left_events]
        right_overstride_mm_vals = [e.get("pelvis_to_landing_ankle_dx_mm_est") for e in right_events]
        overstride_mm_abs = _median_abs(overstride_mm_vals)
        overstride_mm_trimmed = _trimmed_mean(overstride_mm_vals, abs_value=True)
        overstride_selected_mm, _side_mm, _reason_mm = _event_side_representative(left_overstride_mm_vals, right_overstride_mm_vals, prefer_smaller_abs=True)
        overstride_distance_source = "mediapipe_world_dx_low_confidence"

    shank_raw_selected, shank_selected_side, shank_side_reason = _signed_side_representative(
        [e.get("shank_angle_at_contact_deg") for e in left_events],
        [e.get("shank_angle_at_contact_deg") for e in right_events],
    )
    shank_corrected_selected = _corr_shank(shank_raw_selected) if shank_raw_selected not in ("", None) else ""
    if shank_apply_baseline and shank_corrected_selected not in ("", None):
        shank_selected = shank_corrected_selected
        shank_selection_reason = f"baseline_applied_small_offset_{shank_side_reason}"
    else:
        shank_selected = shank_raw_selected
        shank_selection_reason = f"raw_selected_baseline_excluded_{shank_side_reason}" if baseline_num is not None else f"raw_selected_no_baseline_{shank_side_reason}"

    forward_signed = _mean([r.get("forward_lean_deg") for r in frame_rows])
    forward_mm_style = round(abs(float(forward_signed)), 3) if forward_signed not in ("", None) else ""
    actual_fps = _mean([r.get("source_fps") for r in frame_rows])
    try:
        fps_float = float(actual_fps)
    except Exception:
        fps_float = 0.0
    timing_confidence = "low_fps" if fps_float and fps_float < 60 else "medium_fps" if fps_float and fps_float < 100 else "high_fps" if fps_float else "unknown"
    mm_cadence = motionmetrix_values.get("cadence_steps_per_min") or motionmetrix_values.get("cadence_raw_value")
    try:
        expected_event_count_from_mm = round(float(mm_cadence) * duration / 60.0, 3)
    except Exception:
        expected_event_count_from_mm = ""

    summary = {
        "valid_duration_sec": round(duration, 3),
        "valid_frame_count": len(frame_rows),
        "pose_detection_rate": round(sum(1 for r in frame_rows if r.get("pose_detected")) / max(len(frame_rows), 1), 3),
        "actual_video_fps": actual_fps,
        "analysis_fps": actual_fps,
        "timing_confidence": timing_confidence,
        "low_fps_warning": "60fps 미만: Contact Time/Cadence/touch-down 지표 정밀 비교 제한" if fps_float and fps_float < 60 else "",
        "user_input_side_fps": session_meta.get("side_video_fps", ""),
        "user_input_rear_fps": session_meta.get("rear_video_fps", ""),
        "event_count_raw": len(events),
        "event_count_used": len(events_for_summary),
        "edge_adjusted_event_count": edge_adjusted_event_count,
        "cadence_count_spm": cadence_count_spm,
        "cadence_interval_spm": cadence_interval_spm,
        "cadence_edge_adjusted_spm": cadence_edge_spm,
        "cadence_selection_method": cadence_selection_method,
        "expected_event_count_from_mm": expected_event_count_from_mm,
        "event_quality_summary": event_quality_summary,
        "left_step_count": len(left_events),
        "right_step_count": len(right_events),
        "estimated_cadence_spm": cadence_spm,
        "forward_lean_signed_avg_deg": forward_signed,
        "forward_lean_avg_deg": forward_mm_style,
        "left_contact_time_avg_ms": _mean([e.get("contact_time_ms") for e in left_events]),
        "right_contact_time_avg_ms": _mean([e.get("contact_time_ms") for e in right_events]),
        "contact_time_avg_ms": contact_time_ms,
        "contact_time_avg_sec": contact_time_sec,
        "left_overstride_avg_px": _mean([e.get("pelvis_to_landing_ankle_dx_px") for e in left_events]),
        "right_overstride_avg_px": _mean([e.get("pelvis_to_landing_ankle_dx_px") for e in right_events]),
        "overstride_avg_px": _mean([e.get("pelvis_to_landing_ankle_dx_px") for e in events_for_summary]),
        "overstride_avg_abs_px": overstride_px_abs,
        "overstride_trimmed_mean_px": overstride_px_trimmed,
        "left_overstride_avg_mm_est": _mean([e.get("pelvis_to_landing_ankle_dx_mm_est") for e in left_events]),
        "right_overstride_avg_mm_est": _mean([e.get("pelvis_to_landing_ankle_dx_mm_est") for e in right_events]),
        "overstride_avg_mm_est": overstride_mm_abs,
        "overstride_trimmed_mean_mm_est": overstride_mm_trimmed,
        "overstride_selected_mm_est": overstride_selected_mm,
        "overstride_selected_px": overstride_selected_px,
        "overstride_selected_side": overstride_selected_side,
        "overstride_selection_reason": f"{overstride_selection_reason}; distance_source={overstride_distance_source}",
        "overstride_distance_source": overstride_distance_source,
        "left_knee_included_angle_at_contact_avg_deg": _mean([e.get("knee_included_angle_at_contact_deg") for e in left_events]),
        "right_knee_included_angle_at_contact_avg_deg": _mean([e.get("knee_included_angle_at_contact_deg") for e in right_events]),
        "knee_included_angle_at_contact_avg_deg": _mean([e.get("knee_included_angle_at_contact_deg") for e in events_for_summary]),
        "left_knee_flexion_touchdown_avg_deg": left_knee_landing,
        "right_knee_flexion_touchdown_avg_deg": right_knee_landing,
        "knee_flexion_touchdown_avg_deg": _mean([left_knee_landing, right_knee_landing]),
        "left_knee_angle_at_contact_avg_deg": left_knee_landing,
        "right_knee_angle_at_contact_avg_deg": right_knee_landing,
        "knee_angle_at_contact_avg_deg": _mean([left_knee_landing, right_knee_landing]),
        "shank_angle_baseline_offset_deg": shank_baseline,
        "shank_angle_baseline_applied": shank_apply_baseline,
        "left_shank_angle_at_contact_raw_avg_deg": _mean([e.get("shank_angle_at_contact_deg") for e in left_events]),
        "right_shank_angle_at_contact_raw_avg_deg": _mean([e.get("shank_angle_at_contact_deg") for e in right_events]),
        "shank_angle_at_contact_raw_avg_deg": _mean([e.get("shank_angle_at_contact_deg") for e in events_for_summary]),
        "left_shank_angle_at_contact_corrected_avg_deg": _mean([_corr_shank(e.get("shank_angle_at_contact_deg")) for e in left_events]),
        "right_shank_angle_at_contact_corrected_avg_deg": _mean([_corr_shank(e.get("shank_angle_at_contact_deg")) for e in right_events]),
        "shank_angle_at_contact_corrected_avg_deg": _mean([_corr_shank(e.get("shank_angle_at_contact_deg")) for e in events_for_summary]),
        "shank_angle_at_contact_selected_avg_deg": shank_selected,
        "shank_angle_selected_side": shank_selected_side,
        "shank_angle_selection_reason": shank_selection_reason,
        "shank_angle_at_contact_avg_deg": shank_selected,
        "left_foot_angle_at_contact_avg_deg": _mean([e.get("foot_angle_at_contact_deg") for e in left_events]),
        "right_foot_angle_at_contact_avg_deg": _mean([e.get("foot_angle_at_contact_deg") for e in right_events]),
        "foot_angle_at_contact_avg_deg": _mean([e.get("foot_angle_at_contact_deg") for e in events_for_summary]),
        "foot_strike_type_summary": ", ".join(sorted(set(str(e.get("foot_strike_type_estimate", "")) for e in events_for_summary if e.get("foot_strike_type_estimate")))) or "",
        "pelvis_vertical_oscillation_px": vertical_osc_px,
        "pelvis_vertical_oscillation_px_cycle_median": vertical_osc_px_cycle_median,
        "pelvis_vertical_oscillation_px_clip_range": vertical_osc_px_clip_range,
        "body_height_px_median": body_height_px,
        "px_to_mm_scale_est": round(px_to_mm, 6) if px_to_mm else "",
        "scale_source": scale_source,
        "scale_confidence": scale_confidence,
        "x_px_to_mm_scale_est": round(x_px_to_mm, 6) if x_px_to_mm else "",
        "x_scale_source": x_scale_source,
        "x_scale_confidence": x_scale_confidence,
        "y_px_to_mm_scale_est": round(px_to_mm, 6) if px_to_mm else "",
        "y_scale_source": scale_source,
        "y_scale_confidence": scale_confidence,
        "pelvis_vertical_oscillation_mm_est": vertical_osc_mm,
        "pelvis_vertical_displacement_mm_est": vertical_osc_mm,
        "pelvis_forward_deceleration_avg_px_s2_proxy": _mean([e.get("pelvis_forward_deceleration_px_s2_proxy") for e in events_for_summary]),
        "left_max_thigh_flexion_deg": left_thigh_flex,
        "right_max_thigh_flexion_deg": right_thigh_flex,
        "max_thigh_flexion_mean_deg": _mean([left_thigh_flex, right_thigh_flex]),
        "left_max_thigh_extension_deg": left_thigh_ext,
        "right_max_thigh_extension_deg": right_thigh_ext,
        "max_thigh_extension_mean_deg": _mean([left_thigh_ext, right_thigh_ext]),
        "left_max_thigh_extension_magnitude_deg": left_thigh_ext_mag,
        "right_max_thigh_extension_magnitude_deg": right_thigh_ext_mag,
        "max_thigh_extension_magnitude_mean_deg": _mean([left_thigh_ext_mag, right_thigh_ext_mag]),
        "left_max_thigh_flexion_median_neutral_audit_deg": left_thigh_flex_median,
        "right_max_thigh_flexion_median_neutral_audit_deg": right_thigh_flex_median,
        "left_max_thigh_extension_median_neutral_audit_deg": left_thigh_ext_median,
        "right_max_thigh_extension_median_neutral_audit_deg": right_thigh_ext_median,
        "thigh_split_method": "v0.5.11_midrange_neutral_preserve_rom",
        "left_hip_rom_est_deg": left_hip_rom,
        "right_hip_rom_est_deg": right_hip_rom,
        "hip_rom_avg_deg": _mean([left_hip_rom, right_hip_rom]),
        "hip_rom_asymmetry_deg": round(abs(float(left_hip_rom) - float(right_hip_rom)), 3) if left_hip_rom not in ("", None) and right_hip_rom not in ("", None) else "",
        "left_knee_flexion_stance_max_deg": left_knee_stance_max,
        "right_knee_flexion_stance_max_deg": right_knee_stance_max,
        "knee_flexion_stance_max_mean_deg": _mean([left_knee_stance_max, right_knee_stance_max]),
        "left_knee_flexion_swing_max_deg": left_knee_swing_max,
        "right_knee_flexion_swing_max_deg": right_knee_swing_max,
        "knee_flexion_swing_max_mean_deg": _mean([left_knee_swing_max, right_knee_swing_max]),
        "left_knee_rom_est_deg": left_knee_rom,
        "right_knee_rom_est_deg": right_knee_rom,
        "knee_rom_avg_deg": _mean([left_knee_rom, right_knee_rom]),
        "knee_rom_asymmetry_deg": round(abs(float(left_knee_rom) - float(right_knee_rom)), 3) if left_knee_rom not in ("", None) and right_knee_rom not in ("", None) else "",
        "rear_pelvic_tilt_avg_deg": _mean([r.get("rear_pelvic_tilt_deg") for r in frame_rows]),
        "rear_shoulder_tilt_avg_deg": _mean([r.get("rear_shoulder_tilt_deg") for r in frame_rows]),
        "rear_trunk_lateral_tilt_avg_deg": _mean([r.get("rear_trunk_lateral_tilt_deg") for r in frame_rows]),
        "left_knee_alignment_rear_avg_deg": _mean([r.get("left_knee_alignment_rear_deg") for r in frame_rows]),
        "right_knee_alignment_rear_avg_deg": _mean([r.get("right_knee_alignment_rear_deg") for r in frame_rows]),
        "knee_alignment_rear_mean_deg": _mean([r.get("left_knee_alignment_rear_deg") for r in frame_rows] + [r.get("right_knee_alignment_rear_deg") for r in frame_rows]),
        "knee_alignment_rear_abs_mean_deg": _mean([abs(float(v)) for r in frame_rows for v in [r.get("left_knee_alignment_rear_deg"), r.get("right_knee_alignment_rear_deg")] if v not in ("", None)]),
        "knee_medial_collapse_avg_px": _mean([abs(float(v)) for r in frame_rows for v in [r.get("left_knee_medial_offset_px"), r.get("right_knee_medial_offset_px")] if v not in ("", None)]),
        "step_width_avg_px": _median([r.get("step_width_px") for r in frame_rows]),
        "step_width_mean_px_audit": _mean([r.get("step_width_px") for r in frame_rows]),
        "step_width_avg_mm_est": _median([r.get("step_width_mm_est") for r in frame_rows]),
        "step_width_mean_mm_est_audit": _mean([r.get("step_width_mm_est") for r in frame_rows]),
        "crossover_ratio": round(sum(1 for r in frame_rows if r.get("crossover_flag") is True) / max(len(frame_rows), 1), 3),
    }
    for k, v in motionmetrix_values.items():
        if v not in ("", None, []):
            summary[f"motionmetrix_target_{k}"] = v
    return summary


def nearest_event(events: list[dict[str, Any]], timestamp_sec: float) -> dict[str, Any] | None:
    if not events:
        return None
    return min(events, key=lambda e: abs(float(e.get("initial_contact_time_sec", 0) or 0) - timestamp_sec))


def insight_lines(frame_row: dict[str, Any], event: dict[str, Any] | None, view_type: str = "side", max_lines: int = 10) -> list[str]:
    if not frame_row:
        return ["No pose features"]
    lines = [f"Time {frame_row.get('timestamp_sec', '')}s | Frame {frame_row.get('frame_index', '')}"]
    if view_type.startswith("rear") or view_type == "rear":
        items = [
            ("Pelvic Tilt", frame_row.get("rear_pelvic_tilt_deg"), "deg"),
            ("Shoulder Tilt", frame_row.get("rear_shoulder_tilt_deg"), "deg"),
            ("Trunk Tilt", frame_row.get("rear_trunk_lateral_tilt_deg"), "deg"),
            ("Knee Align L", frame_row.get("left_knee_alignment_rear_deg"), "deg"),
            ("Knee Align R", frame_row.get("right_knee_alignment_rear_deg"), "deg"),
            ("Step Width", frame_row.get("step_width_px"), "px"),
            ("Crossover", frame_row.get("crossover_flag"), ""),
        ]
    else:
        items = [
            ("Support", frame_row.get("support_phase"), ""),
            ("Forward Lean", frame_row.get("forward_lean_deg"), "deg"),
            ("Pelvis-Ankle X", frame_row.get("pelvis_to_landing_ankle_dx_px"), "px"),
            ("Knee L/R", f"{frame_row.get('left_knee_angle_deg','')}/{frame_row.get('right_knee_angle_deg','')}", "deg"),
            ("Shank L/R", f"{frame_row.get('left_shank_angle_deg','')}/{frame_row.get('right_shank_angle_deg','')}", "deg"),
            ("Foot Angle L/R", f"{frame_row.get('left_foot_angle_deg','')}/{frame_row.get('right_foot_angle_deg','')}", "deg"),
            ("Pelvis Vx", frame_row.get("pelvis_vx_px_s"), "px/s"),
        ]
    for name, val, unit in items:
        if val not in ("", None):
            lines.append(f"{name}: {val}{(' ' + unit) if unit else ''}")
        if len(lines) >= max_lines:
            break
    if event:
        lines.append(f"IC: {event.get('foot','')} {event.get('initial_contact_time_sec','')}s")
        lines.append(f"Contact: {event.get('contact_time_ms','')} ms")
        lines.append(f"Strike: {event.get('foot_strike_type_estimate','')}")
    return lines[:max_lines]


def direct_target_lines(view_type: str = "side") -> list[str]:
    if view_type == "rear":
        return ["Rear Skeleton-only:"] + [f"- {t}" for t in REAR_SKELETON_FEATURES[:7]]
    return ["MM Target Input:"] + [f"- {t}" for t in SIDE_DIRECT_TARGETS[:7]]


def write_csv(path: Path, rows: list[dict[str, Any]] | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        rows = [rows]
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
