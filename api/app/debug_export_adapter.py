from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import AnalyzeRequest

CANONICAL_PATHS = {
    "meta": "session/session_meta.json",
    "side_frames": "processed/side_running/side_running_all_frame_metrics.csv",
    "side_events": "processed/side_running/side_running_gait_events.csv",
    "side_summary": "processed/side_running/side_running_clip_summary.csv",
    "rear_frames": "processed/rear_running/rear_running_all_frame_metrics.csv",
    "rear_events": "processed/rear_running/rear_running_gait_events.csv",
    "rear_summary": "processed/rear_running/rear_running_clip_summary.csv",
}

TIME_GRID_SEC = np.linspace(-0.250, 0.250, 21).astype(np.float32)
SEQUENCE_CHANNELS = [
    "heel_x_rel_norm",
    "toe_x_rel_norm",
    "ankle_x_rel_norm",
    "heel_ground_gap_norm",
    "toe_ground_gap_norm",
    "ankle_ground_gap_norm",
    "heel_toe_y_diff_norm",
    "foot_angle_canonical_deg",
    "foot_angle_delta_ic_deg",
    "shank_angle_abs_deg",
    "shank_angle_delta_ic_deg",
    "knee_flexion_deg",
    "knee_flexion_delta_ic_deg",
    "heel_gap_velocity_norm_s",
    "toe_gap_velocity_norm_s",
    "heel_toe_velocity_norm_s",
    "foot_angle_velocity_deg_s",
]


@dataclass
class AdapterResult:
    request: AnalyzeRequest
    audit: dict
    source_sha256: str


def _read_json(zf: zipfile.ZipFile, name: str) -> dict:
    if name not in zf.namelist():
        return {}
    return json.loads(zf.read(name).decode("utf-8-sig"))


def _read_csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    if name not in zf.namelist():
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(zf.read(name)), low_memory=False)


def _number(value) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _extract_patient_id(meta: dict) -> str:
    explicit = meta.get("patient_id")
    if explicit is not None and str(explicit).strip():
        text = str(explicit).strip()
        return str(int(text)) if text.isdigit() else text
    candidates = [meta.get("user_id"), meta.get("session_id")]
    for value in candidates:
        if value is None:
            continue
        text = str(value).strip()
        if text.isdigit() and 1 <= len(text) <= 5:
            return str(int(text))
    for value in candidates:
        if value is None:
            continue
        tokens = re.findall(r"(?<!\d)(\d{1,5})(?!\d)", str(value))
        five_digit = [token for token in tokens if len(token) == 5]
        if five_digit:
            return str(int(five_digit[0]))
        if tokens:
            return str(int(tokens[0]))
    raise ValueError("patient_id is missing from session_meta.json")


def _json_number(value):
    x = _number(value)
    return float(x) if np.isfinite(x) else None


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _finite(values) -> pd.Series:
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    return s[np.isfinite(s)]


def _mean(values) -> float:
    s = _finite(values)
    return float(s.mean()) if len(s) else np.nan


def _median(values) -> float:
    s = _finite(values)
    return float(s.median()) if len(s) else np.nan


def _iqr(values) -> float:
    s = _finite(values)
    return float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else np.nan


def _pose_rate(frames: pd.DataFrame) -> float:
    if frames.empty:
        return np.nan
    if "pose_detected" in frames.columns:
        values = frames["pose_detected"]
        if values.dtype == bool:
            return float(values.mean())
        mapped = values.astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
        if mapped.notna().any():
            return float(mapped.mean())
    if "pelvis_center_x" in frames.columns and "pelvis_center_y" in frames.columns:
        valid = _numeric(frames, "pelvis_center_x").notna() & _numeric(frames, "pelvis_center_y").notna()
        return float(valid.mean())
    return np.nan


def _event_count(events: pd.DataFrame, foot: str) -> int:
    if events.empty or "foot" not in events.columns:
        return 0
    return int((events["foot"].astype(str).str.lower() == foot).sum())


def _quality(side_pose, rear_pose, side_l, side_r, rear_l, rear_r) -> str:
    # Side running is the required source for Overstride/Strike. Rear running is
    # optional; when it is absent, do not downgrade otherwise usable side data.
    side_events = min(side_l, side_r)
    rear_available = np.isfinite(rear_pose) and rear_pose > 0 and (rear_l > 0 or rear_r > 0)
    pose_values = [side_pose] if np.isfinite(side_pose) else []
    event_values = [side_events]
    if rear_available:
        pose_values.append(rear_pose)
        event_values.append(min(rear_l, rear_r))
    min_pose = min(pose_values) if pose_values else 0.0
    min_events = min(event_values) if event_values else 0
    if min_pose >= 0.98 and min_events >= 5:
        return "high"
    if min_pose >= 0.90 and min_events >= 3:
        return "medium"
    return "low"


def _nearest(frames: pd.DataFrame, time_sec: float, column: str) -> float:
    if frames.empty or "timestamp_sec" not in frames.columns or column not in frames.columns:
        return np.nan
    t = _numeric(frames, "timestamp_sec")
    v = _numeric(frames, column)
    valid = t.notna() & v.notna()
    if not valid.any():
        return np.nan
    idx = (t[valid] - float(time_sec)).abs().idxmin()
    return float(v.loc[idx])


def _interp(rel_t, values, grid) -> np.ndarray:
    x = np.asarray(rel_t, dtype=float)
    y = np.asarray(values, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    out = np.full(len(grid), np.nan, dtype=np.float32)
    if len(x) < 2:
        return out
    order = np.argsort(x)
    x, y = x[order], y[order]
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]
    if len(xu) < 2:
        return out
    inside = (grid >= xu.min()) & (grid <= xu.max())
    if inside.any():
        out[inside] = np.interp(grid[inside], xu, yu).astype(np.float32)
    return out


def _finite_diff(values, grid) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    x = np.asarray(grid, dtype=float)
    out = np.full(len(y), np.nan, dtype=np.float32)
    m = np.isfinite(y)
    if m.sum() < 3:
        return out
    idx = np.where(m)[0]
    yi = np.interp(np.arange(len(y)), idx, y[idx])
    grad = np.gradient(yi, x)
    out[idx.min():idx.max() + 1] = grad[idx.min():idx.max() + 1].astype(np.float32)
    return out


def _rule_class(value):
    if value is None:
        return None
    s = str(value).strip().lower().replace(" ", "").replace("-", "_")
    for suffix in ["_candidate", "candidate"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)].rstrip("_")
    aliases = {
        "heel": "heel", "rear": "heel", "rearfoot": "heel", "hindfoot": "heel",
        "mid": "midfoot", "midfoot": "midfoot", "middlefoot": "midfoot",
        "fore": "forefoot", "forefoot": "forefoot", "frontfoot": "forefoot",
    }
    return aliases.get(s)


def _extract_sequence(frames: pd.DataFrame, event: pd.Series, foot: str):
    t0 = _number(event.get("initial_contact_time_sec"))
    if frames.empty or foot not in {"left", "right"} or not np.isfinite(t0):
        return None
    required = [
        "timestamp_sec", "ground_y_px",
        f"{foot}_heel_x", f"{foot}_heel_y",
        f"{foot}_toe_x", f"{foot}_toe_y",
        f"{foot}_ankle_x", f"{foot}_ankle_y",
        f"{foot}_shank_angle_deg", f"{foot}_knee_flexion_deg",
    ]
    if any(c not in frames.columns for c in required):
        return None
    ts = pd.to_numeric(frames["timestamp_sec"], errors="coerce")
    w = frames[(ts >= t0 - 0.30) & (ts <= t0 + 0.30)].copy()
    if len(w) < 8:
        return None
    rel = pd.to_numeric(w["timestamp_sec"], errors="coerce").to_numpy(float) - t0
    hx = pd.to_numeric(w[f"{foot}_heel_x"], errors="coerce").to_numpy(float)
    hy = pd.to_numeric(w[f"{foot}_heel_y"], errors="coerce").to_numpy(float)
    tx = pd.to_numeric(w[f"{foot}_toe_x"], errors="coerce").to_numpy(float)
    ty = pd.to_numeric(w[f"{foot}_toe_y"], errors="coerce").to_numpy(float)
    ax = pd.to_numeric(w[f"{foot}_ankle_x"], errors="coerce").to_numpy(float)
    ay = pd.to_numeric(w[f"{foot}_ankle_y"], errors="coerce").to_numpy(float)
    gy = pd.to_numeric(w["ground_y_px"], errors="coerce").to_numpy(float)
    shank = pd.to_numeric(w[f"{foot}_shank_angle_deg"], errors="coerce").to_numpy(float)
    knee = pd.to_numeric(w[f"{foot}_knee_flexion_deg"], errors="coerce").to_numpy(float)
    hx0 = _nearest(frames, t0, f"{foot}_heel_x")
    hy0 = _nearest(frames, t0, f"{foot}_heel_y")
    tx0 = _nearest(frames, t0, f"{foot}_toe_x")
    ty0 = _nearest(frames, t0, f"{foot}_toe_y")
    ax0 = _nearest(frames, t0, f"{foot}_ankle_x")
    if not all(np.isfinite(v) for v in [hx0, hy0, tx0, ty0, ax0]):
        return None
    foot_len0 = math.hypot(tx0 - hx0, ty0 - hy0)
    if not np.isfinite(foot_len0) or foot_len0 < 5:
        return None
    direction = 1.0 if tx0 - hx0 >= 0 else -1.0
    foot_len = np.sqrt((tx - hx) ** 2 + (ty - hy) ** 2)
    foot_len[(~np.isfinite(foot_len)) | (foot_len < 5)] = np.nan
    heel_x_rel = direction * (hx - ax0) / foot_len
    toe_x_rel = direction * (tx - ax0) / foot_len
    ankle_x_rel = direction * (ax - ax0) / foot_len
    heel_gap = (gy - hy) / foot_len
    toe_gap = (gy - ty) / foot_len
    ankle_gap = (gy - ay) / foot_len
    heel_toe_y = (hy - ty) / foot_len
    foot_angle = np.degrees(np.arctan2(-(ty - hy), direction * (tx - hx)))
    seq = {
        "heel_x_rel_norm": _interp(rel, heel_x_rel, TIME_GRID_SEC),
        "toe_x_rel_norm": _interp(rel, toe_x_rel, TIME_GRID_SEC),
        "ankle_x_rel_norm": _interp(rel, ankle_x_rel, TIME_GRID_SEC),
        "heel_ground_gap_norm": _interp(rel, heel_gap, TIME_GRID_SEC),
        "toe_ground_gap_norm": _interp(rel, toe_gap, TIME_GRID_SEC),
        "ankle_ground_gap_norm": _interp(rel, ankle_gap, TIME_GRID_SEC),
        "heel_toe_y_diff_norm": _interp(rel, heel_toe_y, TIME_GRID_SEC),
        "foot_angle_canonical_deg": _interp(rel, foot_angle, TIME_GRID_SEC),
    }
    shank_i = _interp(rel, np.abs(shank), TIME_GRID_SEC)
    knee_i = _interp(rel, knee, TIME_GRID_SEC)
    seq["shank_angle_abs_deg"] = shank_i
    seq["knee_flexion_deg"] = knee_i
    ic = int(np.argmin(np.abs(TIME_GRID_SEC)))
    fa_ic = seq["foot_angle_canonical_deg"][ic]
    sh_ic = shank_i[ic]
    kn_ic = knee_i[ic]
    seq["foot_angle_delta_ic_deg"] = seq["foot_angle_canonical_deg"] - fa_ic if np.isfinite(fa_ic) else np.full(21, np.nan, np.float32)
    seq["shank_angle_delta_ic_deg"] = shank_i - sh_ic if np.isfinite(sh_ic) else np.full(21, np.nan, np.float32)
    seq["knee_flexion_delta_ic_deg"] = knee_i - kn_ic if np.isfinite(kn_ic) else np.full(21, np.nan, np.float32)
    seq["heel_gap_velocity_norm_s"] = _finite_diff(seq["heel_ground_gap_norm"], TIME_GRID_SEC)
    seq["toe_gap_velocity_norm_s"] = _finite_diff(seq["toe_ground_gap_norm"], TIME_GRID_SEC)
    seq["heel_toe_velocity_norm_s"] = _finite_diff(seq["heel_toe_y_diff_norm"], TIME_GRID_SEC)
    seq["foot_angle_velocity_deg_s"] = _finite_diff(seq["foot_angle_canonical_deg"], TIME_GRID_SEC)
    return np.stack([seq[c] for c in SEQUENCE_CHANNELS], axis=1).astype(np.float32)


def _same_foot_cadence(events: pd.DataFrame) -> float:
    if events.empty or "initial_contact_time_sec" not in events.columns or "foot" not in events.columns:
        return np.nan
    e = events.copy()
    e["t"] = pd.to_numeric(e["initial_contact_time_sec"], errors="coerce")
    e = e[e["t"].notna()].sort_values("t")
    values = []
    for foot in ["left", "right"]:
        tt = e.loc[e["foot"].astype(str).str.lower() == foot, "t"].to_numpy(float)
        if len(tt) >= 3:
            d = np.diff(tt)
            d = d[np.isfinite(d) & (d > 0)]
            if len(d):
                values.append(120.0 / np.median(d))
    return float(np.median(values)) if values else np.nan


def _contact_time(events: pd.DataFrame, fps: float) -> float:
    if events.empty:
        return np.nan
    ic = _numeric(events, "initial_contact_time_sec")
    toe = _numeric(events, "toe_off_time_sec")
    values = (toe - ic) * 1000.0
    if np.isfinite(fps) and fps > 0:
        values = values + 1000.0 / fps
    values = values[(values >= 80) & (values <= 800)]
    return _mean(values)


def _thigh_metrics(frames: pd.DataFrame) -> tuple[float, float, float]:
    flexion, extension, hip_rom = [], [], []
    for foot in ["left", "right"]:
        angle = _numeric(frames, f"{foot}_thigh_angle_deg").dropna()
        if len(angle):
            q05 = float(angle.quantile(0.05))
            q95 = float(angle.quantile(0.95))
            flexion.append(q95)
            extension.append(abs(min(q05, 0.0)))
            hip_rom.append(q95 - q05)
    return _mean(flexion), _mean(extension), _mean(hip_rom)


def _knee_touchdown(events: pd.DataFrame) -> float:
    return _mean(_numeric(events, "knee_flexion_at_contact_deg"))


def _pelvic_metrics(frames: pd.DataFrame, events: pd.DataFrame) -> tuple[float, float]:
    if frames.empty or events.empty or "timestamp_sec" not in frames.columns or "rear_pelvic_tilt_deg" not in frames.columns:
        return np.nan, np.nan
    t = _numeric(frames, "timestamp_sec")
    tilt = _numeric(frames, "rear_pelvic_tilt_deg")
    per_foot = {"left": [], "right": []}
    absmax = []
    for _, event in events.iterrows():
        foot = str(event.get("foot", "")).lower()
        if foot not in per_foot:
            continue
        ic = _number(event.get("initial_contact_time_sec"))
        toe = _number(event.get("toe_off_time_sec"))
        if not np.isfinite(ic) or not np.isfinite(toe) or toe <= ic:
            continue
        stance = tilt[(t >= ic) & (t <= toe) & tilt.notna()]
        if len(stance) < 2:
            continue
        peak_idx = stance.abs().idxmax()
        per_foot[foot].append(float(stance.loc[peak_idx]))
        absmax.append(float(stance.abs().max()))
    left = _median(per_foot["left"])
    right = _median(per_foot["right"])
    pelvic_drop = _median(absmax)
    hip_hike = abs(left - right) if np.isfinite(left) and np.isfinite(right) else np.nan
    return pelvic_drop, hip_hike


def _package_bytes(data: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        if CANONICAL_PATHS["meta"] in names:
            return data
        nested = [n for n in zf.namelist() if n.lower().endswith(".zip")]
        if len(nested) == 1:
            return zf.read(nested[0])
        raise ValueError("Debug Export ZIP must contain one patient package.")


def _sequence_json(arr: np.ndarray):
    return [[float(x) if np.isfinite(x) else None for x in row] for row in arr]


class DebugExportAdapter:
    def __init__(self, data: bytes):
        self.original_sha256 = hashlib.sha256(data).hexdigest()
        self.data = _package_bytes(data)

    @classmethod
    def from_path(cls, path: str | Path):
        return cls(Path(path).read_bytes())

    @classmethod
    def from_bytes(cls, data: bytes):
        return cls(data)

    def build(self) -> AdapterResult:
        with zipfile.ZipFile(io.BytesIO(self.data)) as zf:
            names = set(zf.namelist())
            required = [CANONICAL_PATHS[k] for k in ["meta", "side_frames", "side_events", "side_summary"]]
            missing = [p for p in required if p not in names]
            if missing:
                raise ValueError("Missing Debug Export files: " + ", ".join(missing))
            meta = _read_json(zf, CANONICAL_PATHS["meta"])
            side_frames = _read_csv(zf, CANONICAL_PATHS["side_frames"])
            side_events = _read_csv(zf, CANONICAL_PATHS["side_events"])
            side_summary = _read_csv(zf, CANONICAL_PATHS["side_summary"])
            rear_frames = _read_csv(zf, CANONICAL_PATHS["rear_frames"])
            rear_events = _read_csv(zf, CANONICAL_PATHS["rear_events"])
            rear_summary = _read_csv(zf, CANONICAL_PATHS["rear_summary"])

        summary = side_summary.iloc[0] if len(side_summary) else pd.Series(dtype=object)
        rear_summary_row = rear_summary.iloc[0] if len(rear_summary) else pd.Series(dtype=object)
        patient_id = _extract_patient_id(meta)
        session_id = str(meta.get("session_id") or patient_id)
        height = _number(meta.get("height_cm"))
        weight = _number(meta.get("weight_kg"))
        speed = _number(meta.get("running_speed_kmh"))
        fps = _number(summary.get("actual_video_fps"))
        if not np.isfinite(fps) and "source_fps" in side_events.columns:
            fps = _median(side_events["source_fps"])
        rear_fps = _number(rear_summary_row.get("actual_video_fps"))
        if not np.isfinite(rear_fps) and "source_fps" in rear_events.columns:
            rear_fps = _median(rear_events["source_fps"])

        side_pose = _pose_rate(side_frames)
        rear_pose = _pose_rate(rear_frames)
        side_l = _event_count(side_events, "left")
        side_r = _event_count(side_events, "right")
        rear_l = _event_count(rear_events, "left")
        rear_r = _event_count(rear_events, "right")
        quality = _quality(side_pose, rear_pose, side_l, side_r, rear_l, rear_r)

        event_dx = _numeric(side_events, "pelvis_to_landing_ankle_dx_mm_est")
        left_events = side_events[side_events["foot"].astype(str).str.lower() == "left"] if "foot" in side_events.columns else pd.DataFrame()
        right_events = side_events[side_events["foot"].astype(str).str.lower() == "right"] if "foot" in side_events.columns else pd.DataFrame()
        left_dx = _numeric(left_events, "pelvis_to_landing_ankle_dx_mm_est")
        right_dx = _numeric(right_events, "pelvis_to_landing_ankle_dx_mm_est")
        left_med = _median(left_dx.abs())
        right_med = _median(right_dx.abs())
        foot_angle = _numeric(side_events, "foot_angle_at_contact_deg")
        shank = _numeric(side_events, "shank_angle_at_contact_deg")
        knee = _numeric(side_events, "knee_flexion_at_contact_deg")
        contact = _numeric(side_events, "contact_time_ms")
        thigh_flexion, thigh_extension, hip_rom_recalc = _thigh_metrics(side_frames)

        overstride_features = {
            "os_event_abs_mean_mm": _json_number(_mean(event_dx.abs())),
            "os_event_abs_median_mm": _json_number(_median(event_dx.abs())),
            "os_event_abs_iqr_mm": _json_number(_iqr(event_dx.abs())),
            "os_event_signed_mean_mm": _json_number(_mean(event_dx)),
            "os_left_abs_median_mm": _json_number(left_med),
            "os_right_abs_median_mm": _json_number(right_med),
            "os_lr_abs_diff_mm": _json_number(abs(left_med - right_med) if np.isfinite(left_med) and np.isfinite(right_med) else np.nan),
            "os_clip_avg_mm": _json_number(summary.get("overstride_avg_mm_est")),
            "os_clip_trimmed_mean_mm": _json_number(summary.get("overstride_trimmed_mean_mm_est")),
            "os_clip_selected_mm": _json_number(summary.get("overstride_selected_mm_est")),
            "foot_angle_abs_median_deg": _json_number(_median(foot_angle.abs())),
            "foot_angle_iqr_deg": _json_number(_iqr(foot_angle)),
            "shank_angle_abs_median_deg": _json_number(_median(shank.abs())),
            "shank_angle_median_deg": _json_number(_median(shank)),

            "knee_landing_event_median_deg": _json_number(_median(knee)),
            "thigh_flexion_recalc_deg": _json_number(thigh_flexion),
            "hip_rom_recalc_deg": _json_number(hip_rom_recalc),
            "contact_time_event_median_ms": _json_number(_median(contact)),
            "height_cm": _json_number(height),
            "running_speed_kmh": _json_number(speed),
            "side_pose_rate": _json_number(side_pose),
        }

        strike_feet = []
        sequence_counts = {}
        for foot in ["left", "right"]:
            events = side_events[side_events["foot"].astype(str).str.lower() == foot] if "foot" in side_events.columns else pd.DataFrame()
            rows = []
            for _, event in events.iterrows():
                arr = _extract_sequence(side_frames, event, foot)
                if arr is None:
                    continue
                rows.append({
                    "sequence": _sequence_json(arr),
                    "rule_class": _rule_class(event.get("foot_strike_type_estimate")),
                })
            if not rows:
                raise ValueError(f"No usable Strike sequence for {foot} foot.")
            strike_feet.append({"foot": foot, "events": rows})
            sequence_counts[foot] = len(rows)

        cadence = _same_foot_cadence(side_events)
        contact_time = _contact_time(side_events, fps)
        forward_lean = _mean(_numeric(side_frames, "forward_lean_deg"))
        knee_touchdown = _knee_touchdown(side_events)
        shank_touchdown = _median(shank)
        pelvic_drop, hip_hike = _pelvic_metrics(rear_frames, rear_events)

        posture_metrics = {
            "cadence_spm": _json_number(cadence),
            "contact_time_ms": _json_number(contact_time),
            "forward_lean_deg": _json_number(forward_lean),
            "max_thigh_flexion_deg": _json_number(thigh_flexion),
            "max_thigh_extension_deg": _json_number(thigh_extension),
            "hip_rom_deg": _json_number(hip_rom_recalc),
            "knee_flexion_touchdown_deg": _json_number(knee_touchdown),
            "max_knee_flexion_swing_deg": _json_number(summary.get("knee_flexion_swing_max_mean_deg")),
            "knee_rom_deg": _json_number(summary.get("knee_rom_avg_deg")),
            "vertical_displacement_mm": _json_number(summary.get("pelvis_vertical_displacement_mm_est")),
            "pelvic_drop_deg": _json_number(pelvic_drop),
            "hip_hike_difference_deg": _json_number(hip_hike),
            "shank_angle_touchdown_deg": _json_number(shank_touchdown),
        }

        request = AnalyzeRequest.model_validate({
            "session_id": session_id,
            "patient_meta": {
                "patient_id": patient_id,
                "height_cm": _json_number(height),
                "weight_kg": _json_number(weight),
                "running_speed_kmh": _json_number(speed),
                "video_fps": _json_number(fps),
            },
            "overstride_features": overstride_features,
            "strike_feet": strike_feet,
            "posture_metrics": posture_metrics,
            "quality": {
                "quality_tier": quality,
                "side_pose_rate": _json_number(side_pose),
                "rear_pose_rate": _json_number(rear_pose),
                "status": "review_required" if quality == "low" else "completed",
                "unmeasurable_reasons": [],
                "notes": [],
            },
        })

        audit = {
            "patient_id": patient_id,
            "session_id": session_id,
            "side_actual_fps": _json_number(fps),
            "rear_actual_fps": _json_number(rear_fps),
            "side_pose_rate": _json_number(side_pose),
            "rear_pose_rate": _json_number(rear_pose),
            "side_event_left": side_l,
            "side_event_right": side_r,
            "rear_event_left": rear_l,
            "rear_event_right": rear_r,
            "quality_tier": quality,
            "cadence_same_foot_spm": _json_number(cadence),
            "contact_time_inclusive_mean_ms": _json_number(contact_time),
            "forward_lean_recalc_deg": _json_number(forward_lean),
            "thigh_flexion_recalc_deg": _json_number(thigh_flexion),
            "thigh_extension_recalc_deg": _json_number(thigh_extension),
            "hip_rom_recalc_deg": _json_number(hip_rom_recalc),
            "knee_landing_event_recalc_deg": _json_number(knee_touchdown),
            "shank_angle_median_deg": _json_number(shank_touchdown),
            "pelvic_drop_descriptive_deg": _json_number(pelvic_drop),
            "hip_hike_difference_deg": _json_number(hip_hike),
            "strike_event_left": sequence_counts["left"],
            "strike_event_right": sequence_counts["right"],
            **overstride_features,
        }
        return AdapterResult(request=request, audit=audit, source_sha256=self.original_sha256)
