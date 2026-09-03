from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .storage import BASE_DIR, session_path, write_json, read_json
from .session_exports import infer_video_role_from_name
from .gait_features import (
    build_clip_summary,
    build_second_summary,
    compute_frame_metrics,
    detect_gait_events,
    direct_target_lines,
    infer_contacts,
    insight_lines,
    nearest_event,
    write_csv,
)

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

MP_POSE = None
MP_IMPORT_ERROR = None
MP_VERSION = None
if mp is not None:
    MP_VERSION = getattr(mp, "__version__", "unknown")
    try:
        MP_POSE = mp.solutions.pose
    except Exception as exc:
        try:
            from mediapipe.python.solutions import pose as MP_POSE
        except Exception as fallback_exc:
            MP_POSE = None
            MP_IMPORT_ERROR = (
                f"현재 mediapipe {MP_VERSION}에서 legacy mp.solutions.pose를 찾지 못했습니다. "
                "mediapipe 0.10.31 이상에서는 기존 Solutions API가 제거되어 Preview가 동작하지 않습니다. "
                "requirements.txt 기준으로 mediapipe==0.10.21을 재설치해 주세요. "
                f"original: {exc}; fallback: {fallback_exc}"
            )
else:
    MP_IMPORT_ERROR = "mediapipe is not installed"

MP_POSE_IDS = {7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
POSE_EDGES = [
    (7, 11), (8, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

WORKFLOW_STEPS = [
    {
        "step_id": "rear_static",
        "title": "1. 후면 전신 정지 2~3초",
        "slot": "rear_static",
        "view": "rear",
        "guide": "후면 중앙에서 전신이 보이도록 세우고 골반 23·24, 발목 27·28, 몸통 중심선을 확인합니다.",
        "metrics": ["pelvic_drop", "trunk_lateral_tilt"],
    },
    {
        "step_id": "rear_running",
        "title": "2. 후면 달리기 촬영/확인",
        "slot": "rear_running",
        "view": "rear",
        "guide": "골반 낙하, 무릎 안쪽 붕괴, 스텝 폭/크로스오버, 몸통 좌우 기울기를 확인합니다.",
        "metrics": ["pelvic_drop", "knee_medial_collapse", "step_width_crossover", "trunk_lateral_tilt"],
    },
    {
        "step_id": "side_static",
        "title": "3. 측면 전신 정지 2~3초",
        "slot": "side_static",
        "view": "side",
        "guide": "측면에서 어깨 중심, 골반 중심, 수직 기준선, 지면선을 확인합니다.",
        "metrics": ["forward_lean_deg"],
    },
    {
        "step_id": "side_running",
        "title": "4. 측면 달리기 촬영/확인",
        "slot": "side_running",
        "view": "side",
        "guide": "Forward Lean, Overstride, Shank Angle, Contact Time 등 측면 지표를 확인합니다.",
        "metrics": ["forward_lean_deg", "overstride", "braking_force", "hip_thigh_flexion_extension", "knee_flexion_rom", "shank_angle", "foot_strike_type", "cadence", "contact_time", "vertical_oscillation", "step_stride_length", "flight_time"],
    },
]


def python_runtime_status() -> dict[str, Any]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    preview_supported = (sys.version_info.major == 3 and 9 <= sys.version_info.minor <= 12)
    return {"version": version, "preview_supported": preview_supported}


def dependencies_status() -> dict[str, bool | str]:
    py = python_runtime_status()
    return {
        "python_version": py["version"],
        "python_preview_supported": py["preview_supported"],
        "opencv": cv2 is not None,
        "mediapipe": MP_POSE is not None,
        "mediapipe_version": MP_VERSION or "not_installed",
    }


def dependency_message() -> str:
    py = python_runtime_status()
    if not py["preview_supported"]:
        return (
            f"현재 Python {py['version']} 환경입니다. Skeleton Preview는 MediaPipe 호환성을 위해 "
            "Python 3.9~3.12 환경을 권장합니다. Windows에서는 setup_windows.bat로 전용 가상환경을 생성해 주세요."
        )
    if cv2 is None and MP_POSE is None:
        return "OpenCV와 MediaPipe Pose backend를 사용할 수 없습니다. requirements.txt 기준으로 재설치해 주세요."
    if cv2 is None:
        return "OpenCV를 사용할 수 없습니다. opencv-python-headless를 재설치해 주세요."
    if MP_POSE is None:
        return (
            "MediaPipe Pose backend를 사용할 수 없습니다. "
            "현재 환경에 설치된 mediapipe가 legacy mp.solutions.pose를 제공하지 않는 버전일 수 있습니다. "
            "setup_windows.bat를 다시 실행하거나 아래 명령으로 재설치해 주세요: "
            "`.venv\\Scripts\\python.exe -m pip install --force-reinstall mediapipe==0.10.21 numpy<2 protobuf<5` "
            f"({MP_IMPORT_ERROR})"
        )
    return ""


def _to_pil(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    if arr.shape[-1] == 4:
        return Image.fromarray(arr[:, :, :3]).convert("RGB")
    return Image.fromarray(arr).convert("RGB")


def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def extract_frame(video_path: Path, timestamp_sec: float) -> Image.Image:
    if cv2 is None:
        raise RuntimeError("OpenCV가 설치되어 있지 않습니다.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다.")
    
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        try:
            cap.set(orientation_auto, 1)
        except Exception:
            pass
    
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    frame_index = max(0, int(timestamp_sec * source_fps))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("해당 시점의 프레임을 읽지 못했습니다.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def _detect_landmarks(image: Image.Image, pose_backend: Any | None = None) -> dict[int, dict[str, float]]:
    """Detect MediaPipe landmarks and keep both image and world coordinates.

    image x/y are used for drawing and fallback calculations. world_x/y/z are
    MediaPipe's estimated 3D landmarks, used in v0.5.5 for MotionMetrix-definition-aligned angle/ROM and mm
    estimate columns where available. They are not treated as MotionMetrix-grade
    calibrated depth measurements.

    A caller may pass a reusable Pose backend. This keeps frame-level inference
    identical while avoiding repeated graph/model initialization for every frame.
    """
    if MP_POSE is None:
        return {}
    arr = _pil_to_rgb_array(image)
    if pose_backend is None:
        with MP_POSE.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            result = pose.process(arr)
    else:
        result = pose_backend.process(arr)
    if not result.pose_landmarks:
        return {}
    h, w = arr.shape[:2]
    points = {}
    world_landmarks = getattr(result, "pose_world_landmarks", None)
    for idx, lm in enumerate(result.pose_landmarks.landmark):
        if idx not in MP_POSE_IDS:
            continue
        item = {"x": float(lm.x * w), "y": float(lm.y * h), "visibility": float(getattr(lm, "visibility", 0.0))}
        if world_landmarks and idx < len(world_landmarks.landmark):
            wlm = world_landmarks.landmark[idx]
            item.update({
                "world_x": float(wlm.x),
                "world_y": float(wlm.y),
                "world_z": float(wlm.z),
            })
        points[idx] = item
    return points

def _midpoint(points: dict[int, dict[str, float]], a: int, b: int):
    if a not in points or b not in points:
        return None
    return ((points[a]["x"] + points[b]["x"]) / 2, (points[a]["y"] + points[b]["y"]) / 2)


def _angle_between(p1, p2, p3) -> float | None:
    if not p1 or not p2 or not p3:
        return None
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return None
    cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))


def _line_angle_to_vertical(p_low, p_high) -> float | None:
    if not p_low or not p_high:
        return None
    dx = p_high[0] - p_low[0]
    dy = p_low[1] - p_high[1]
    if dy == 0 and dx == 0:
        return None
    return float(math.degrees(math.atan2(dx, dy)))


def _point_tuple(points, idx):
    if idx not in points:
        return None
    return (points[idx]["x"], points[idx]["y"])


def _fmt(v, unit=""):
    if v is None:
        return "N/A"
    return f"{v:.1f}{unit}"


def compute_reference_values(points: dict[int, dict[str, float]], metric_id: str) -> dict[str, str]:
    shoulder = _midpoint(points, 11, 12)
    pelvis = _midpoint(points, 23, 24)
    values: dict[str, str] = {}

    if metric_id == "forward_lean_deg":
        values["Forward Lean"] = _fmt(_line_angle_to_vertical(pelvis, shoulder), " deg")
        values["Basis"] = "pelvis center to shoulder center vs vertical axis"

    elif metric_id == "trunk_lateral_tilt":
        values["Trunk lateral tilt"] = _fmt(_line_angle_to_vertical(pelvis, shoulder), " deg")
        values["Basis"] = "shoulder center to pelvis center line in rear view"

    elif metric_id == "overstride":
        if pelvis and 27 in points and 28 in points:
            ankle_id = 27 if points[27]["y"] >= points[28]["y"] else 28
            px = points[ankle_id]["x"] - pelvis[0]
            values["Landing ankle candidate"] = f"point {ankle_id}"
            values["Pelvis-ankle horizontal distance"] = _fmt(px, " px")
            values["Note"] = "cm conversion requires scale calibration in phase 3"

    elif metric_id == "knee_flexion_rom":
        l = _angle_between(_point_tuple(points, 23), _point_tuple(points, 25), _point_tuple(points, 27))
        r = _angle_between(_point_tuple(points, 24), _point_tuple(points, 26), _point_tuple(points, 28))
        lf = 180 - l if l is not None else None
        rf = 180 - r if r is not None else None
        values["Left knee flexion"] = _fmt(lf, " deg")
        values["Right knee flexion"] = _fmt(rf, " deg")
        values["Note"] = "MotionMetrix comparison uses flexion angle, not included joint angle"

    elif metric_id == "hip_thigh_flexion_extension":
        if 23 in points and 25 in points:
            values["Left thigh vs vertical"] = _fmt(_line_angle_to_vertical(_point_tuple(points, 23), _point_tuple(points, 25)), " deg")
        if 24 in points and 26 in points:
            values["Right thigh vs vertical"] = _fmt(_line_angle_to_vertical(_point_tuple(points, 24), _point_tuple(points, 26)), " deg")

    elif metric_id == "shank_angle":
        values["Left shank angle"] = _fmt(_line_angle_to_vertical(_point_tuple(points, 27), _point_tuple(points, 25)), " deg")
        values["Right shank angle"] = _fmt(_line_angle_to_vertical(_point_tuple(points, 28), _point_tuple(points, 26)), " deg")

    elif metric_id == "pelvic_drop":
        if 23 in points and 24 in points:
            values["Pelvis line tilt"] = _fmt(_line_angle_to_vertical(_point_tuple(points, 23), _point_tuple(points, 24)), " deg")
            values["Basis"] = "left hip point 23 to right hip point 24"

    elif metric_id == "knee_medial_collapse":
        l = _angle_between(_point_tuple(points, 23), _point_tuple(points, 25), _point_tuple(points, 27))
        r = _angle_between(_point_tuple(points, 24), _point_tuple(points, 26), _point_tuple(points, 28))
        values["Left hip-knee-ankle alignment"] = _fmt(l, " deg")
        values["Right hip-knee-ankle alignment"] = _fmt(r, " deg")

    elif metric_id == "step_width_crossover":
        if 27 in points and 28 in points:
            values["Ankle horizontal distance"] = _fmt(abs(points[27]["x"] - points[28]["x"]), " px")
            values["Note"] = "crossover decision requires segment/centerline review"

    elif metric_id == "foot_strike_type":
        if 29 in points and 31 in points:
            values["Left heel-toe height diff"] = _fmt(points[29]["y"] - points[31]["y"], " px")
        if 30 in points and 32 in points:
            values["Right heel-toe height diff"] = _fmt(points[30]["y"] - points[32]["y"], " px")
        values["Note"] = "foot strike type is saved as manual input"

    elif metric_id in {"contact_time", "cadence", "flight_time", "vertical_oscillation", "step_stride_length", "braking_force"}:
        values["Note"] = "event/segment-based metric; current frame is for point verification"

    detected = len([k for k, v in points.items() if v.get("visibility", 0) >= 0.5])
    values["Detected points"] = f"{detected}/{len(MP_POSE_IDS)}"
    return values

def _draw_info_panel(draw: ImageDraw.ImageDraw, image: Image.Image, lines: list[str], xy=(10, 10), title_fill=(30, 60, 120), fill=(255, 255, 255), max_width: int | None = None) -> int:
    """Draw compact English overlay panel and return bottom y."""
    if not lines:
        return xy[1]
    x, y = xy
    max_width = max_width or min(image.width - x - 10, 520)
    line_h = 22
    width = min(max_width, max(280, max(len(str(line)) for line in lines) * 8 + 28))
    height = line_h * len(lines) + 14
    draw.rounded_rectangle((x, y, x + width, y + height), radius=8, fill=fill, outline=(70, 70, 70), width=1)
    cy = y + 8
    for idx, line in enumerate(lines):
        color = title_fill if idx == 0 else (0, 0, 0)
        draw.text((x + 12, cy), str(line)[:70], fill=color)
        cy += line_h
    return y + height


def _draw_pose_overlay_from_points(
    image: Image.Image,
    points: dict[int, dict[str, float]],
    metric: dict[str, Any] | None = None,
    show_all: bool = False,
    stats: dict[str, Any] | None = None,
    feature_row: dict[str, Any] | None = None,
    event: dict[str, Any] | None = None,
    view_type: str = "side",
    overlay_mode: str = "summary",
) -> Image.Image:
    metric = metric or {}
    active_points = {int(p) for p in metric.get("required_points", []) if str(p).isdigit()}
    image = _to_pil(image)
    draw = ImageDraw.Draw(image)

    if not points:
        draw.rounded_rectangle((10, 10, min(image.width - 10, 680), 72), radius=8, fill=(255, 255, 255))
        draw.text((18, 22), "Pose landmark detection failed", fill=(0, 0, 0))
        draw.text((18, 46), "Check full-body visibility, lighting, and frame timing.", fill=(0, 0, 0))
        return image

    visible_points = set(points) if show_all or not active_points else active_points
    for a, b in POSE_EDGES:
        if a in points and b in points and (show_all or a in visible_points or b in visible_points):
            color = (60, 180, 120) if (a in active_points or b in active_points) else (150, 150, 150)
            draw.line([(points[a]["x"], points[a]["y"]), (points[b]["x"], points[b]["y"])], fill=color, width=4 if color[1] == 180 else 2)

    shoulder = _midpoint(points, 11, 12)
    pelvis = _midpoint(points, 23, 24)
    if shoulder and pelvis:
        draw.line([pelvis, shoulder], fill=(255, 140, 0), width=4)
        draw.line([(pelvis[0], 0), (pelvis[0], image.height)], fill=(80, 160, 255), width=2)
    foot_ys = [points[i]["y"] for i in [29, 30, 31, 32] if i in points]
    ground_y = None
    if feature_row and feature_row.get("ground_y_px") not in (None, ""):
        try:
            ground_y = float(feature_row.get("ground_y_px"))
        except Exception:
            ground_y = None
    if ground_y is None and foot_ys:
        ground_y = max(foot_ys)
    if ground_y is not None:
        draw.line([(0, ground_y), (image.width, ground_y)], fill=(255, 80, 80), width=2)

    # Calculation guide lines for intuitive review.
    if pelvis and feature_row:
        ankle_x = feature_row.get("landing_ankle_x")
        ankle_y = feature_row.get("landing_ankle_y")
        if ankle_x not in (None, "") and ankle_y not in (None, ""):
            try:
                ax, ay = float(ankle_x), float(ankle_y)
                draw.line([(pelvis[0], ay), (ax, ay)], fill=(255, 215, 0), width=4)
                draw.polygon([(ax, ay), (ax - 8, ay - 5), (ax - 8, ay + 5)], fill=(255, 215, 0))
                draw.text((min(pelvis[0], ax) + 8, ay - 24), "Pelvis-Ankle X", fill=(0, 0, 0))
            except Exception:
                pass
    # Knee/shank/foot emphasis lines.
    for foot, ids in {"left": (23, 25, 27, 29, 31), "right": (24, 26, 28, 30, 32)}.items():
        hip, knee, ankle, heel, toe = ids
        if hip in points and knee in points and ankle in points:
            draw.line([(points[hip]["x"], points[hip]["y"]), (points[knee]["x"], points[knee]["y"]), (points[ankle]["x"], points[ankle]["y"])], fill=(0, 210, 255), width=2)
        if heel in points and toe in points:
            draw.line([(points[heel]["x"], points[heel]["y"]), (points[toe]["x"], points[toe]["y"])], fill=(255, 120, 0), width=3)

    for idx, p in points.items():
        if idx not in visible_points:
            continue
        active = idx in active_points
        r = 7 if active else 5
        fill_color = (255, 200, 0) if active else (235, 235, 235)
        x, y = p["x"], p["y"]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill_color, outline=(0, 0, 0), width=2)
        draw.text((x + 8, y - 8), str(idx), fill=(0, 0, 0))

    title = metric.get("display_name_en") or metric.get("metric_id", "Skeleton Preview")
    if overlay_mode == "detail" and stats:
        lines = [title] + [f"{k}: {v}" for k, v in list(stats.items())[:7]]
    else:
        lines = ["Skeleton Metric Insight"] + insight_lines(feature_row or {}, event, view_type=view_type, max_lines=10)
    bottom = _draw_info_panel(draw, image, lines, xy=(10, 10), title_fill=(20, 80, 170), fill=(255, 255, 255,))
    target = direct_target_lines("rear" if str(view_type).startswith("rear") else "side")
    _draw_info_panel(draw, image, target, xy=(10, bottom + 8), title_fill=(220, 100, 0), fill=(255, 245, 230), max_width=420)
    return image


def overlay_pose(
    image: Image.Image,
    metric: dict[str, Any] | None = None,
    show_all: bool = False,
    view_type: str = "side",
    frame_index: int = 0,
    timestamp_sec: float = 0.0,
    source_fps: float = 30.0,
    overlay_mode: str = "detail",
) -> tuple[Image.Image, dict[str, Any]]:
    metric = metric or {}
    image = _to_pil(image)
    draw = ImageDraw.Draw(image)
    if MP_POSE is None:
        msg = dependency_message()
        draw.rounded_rectangle((10, 10, min(image.width - 10, 820), 88), radius=8, fill=(255, 255, 255))
        draw.text((18, 22), "MediaPipe Pose backend unavailable", fill=(0, 0, 0))
        draw.text((18, 46), "Use Python 3.9-3.12 venv and run setup_windows.bat", fill=(0, 0, 0))
        draw.text((18, 68), "Saved video upload and manual labeling still work.", fill=(0, 0, 0))
        return image, {"Status": "MediaPipe Pose backend unavailable", "Action": "Run setup_windows.bat or reinstall dependencies from requirements.txt", "Detail": msg}
    points = _detect_landmarks(image)
    if not points:
        return _draw_pose_overlay_from_points(image, {}, metric=metric, show_all=show_all), {"Status": "Pose landmark detection failed"}
    stats: dict[str, Any] = compute_reference_values(points, metric.get("metric_id", ""))
    feature_row = compute_frame_metrics(points, frame_index=frame_index, timestamp_sec=timestamp_sec, source_fps=source_fps, view_type=view_type)
    stats.update({k: v for k, v in feature_row.items() if k not in stats})
    image = _draw_pose_overlay_from_points(image, points, metric=metric, show_all=show_all, stats=stats, feature_row=feature_row, event=None, view_type=view_type, overlay_mode=overlay_mode)
    return image, stats

def _safe_slug(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(value))
    return cleaned.strip("_") or "overlay"


def _transcode_to_browser_mp4(input_video: Path, output_video: Path) -> dict[str, Any]:
    """Transcode OpenCV output to browser-friendly H.264 MP4.

    OpenCV often writes MP4 using the mp4v codec. The file may download and
    play in desktop players, but browser players used by Streamlit/Chrome can
    show a black player or 0:00 duration. Using imageio-ffmpeg gives us a
    bundled ffmpeg binary on Streamlit Cloud and creates yuv420p H.264 MP4.
    """
    info: dict[str, Any] = {
        "transcoded_for_browser": False,
        "browser_video_codec": "opencv_mp4v_fallback",
        "transcode_message": "ffmpeg transcoding was not run",
    }
    ffmpeg_exe = None
    if imageio_ffmpeg is not None:
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            info["transcode_message"] = f"imageio-ffmpeg unavailable: {exc}"

    if not ffmpeg_exe:
        shutil.copyfile(input_video, output_video)
        return info

    tmp_out = output_video.with_suffix(".h264.tmp.mp4")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-movflags",
        "+faststart",
        str(tmp_out),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 0:
            tmp_out.replace(output_video)
            info.update(
                {
                    "transcoded_for_browser": True,
                    "browser_video_codec": "h264_yuv420p",
                    "transcode_message": "Converted to browser-friendly H.264 MP4",
                }
            )
            return info
        info["transcode_message"] = (proc.stderr or proc.stdout or "ffmpeg failed").strip()[:500]
    except Exception as exc:
        info["transcode_message"] = f"ffmpeg exception: {exc}"
    finally:
        try:
            tmp_out.unlink(missing_ok=True)
        except Exception:
            pass

    shutil.copyfile(input_video, output_video)
    return info




def _even_video_size(width: int, height: int) -> tuple[int, int]:
    """Return even dimensions required by H.264 yuv420p encoders."""
    safe_w = int(width) - (int(width) % 2)
    safe_h = int(height) - (int(height) % 2)
    return max(2, safe_w), max(2, safe_h)


def _crop_rgb_to_size(frame_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Crop RGB array to the exact encoder size without resizing artifacts."""
    return np.ascontiguousarray(frame_rgb[:height, :width, :3])


def _open_browser_h264_pipe(output_video: Path, width: int, height: int, fps: float):
    """Open an ffmpeg pipe that writes browser-playable H.264 MP4 directly.

    This avoids the common Streamlit/Chrome black-player issue caused by
    OpenCV's mp4v output or partially browser-compatible MP4 metadata.
    """
    if imageio_ffmpeg is None:
        return None, "imageio-ffmpeg is not installed"
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        return None, f"imageio-ffmpeg unavailable: {exc}"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc, "ffmpeg H.264 pipe opened"
    except Exception as exc:
        return None, f"ffmpeg pipe open failed: {exc}"

def _prepare_gif_frame(image: Image.Image, max_width: int = 720) -> Image.Image:
    """Create a browser-safe animated preview frame.

    Streamlit's in-page MP4 player can fail depending on browser codec support,
    while the downloaded MP4 is still valid. A compact GIF preview avoids that
    codec path and gives customers an immediately visible result in the app.
    """
    frame = image.convert("RGB")
    if frame.width > max_width:
        ratio = max_width / float(frame.width)
        new_size = (max_width, max(1, int(frame.height * ratio)))
        frame = frame.resize(new_size, Image.Resampling.LANCZOS)
    return frame


def _save_browser_preview_gif(frames: list[Image.Image], output_gif: Path, output_fps: float) -> dict[str, Any]:
    info: dict[str, Any] = {
        "browser_preview_type": "gif",
        "browser_preview_path": "",
        "browser_preview_message": "GIF preview was not created",
        "browser_preview_frames": 0,
    }
    if not frames:
        return info
    duration_ms = int(round(1000 / max(1.0, min(float(output_fps or 8.0), 12.0))))
    try:
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        if output_gif.exists() and output_gif.stat().st_size > 0:
            info.update(
                {
                    "browser_preview_path": str(output_gif.relative_to(BASE_DIR)),
                    "browser_preview_message": "Browser preview GIF created",
                    "browser_preview_frames": len(frames),
                }
            )
    except Exception as exc:
        info["browser_preview_message"] = f"GIF preview failed: {exc}"
    return info


def _infer_view_type(video_path: Path, metric: dict[str, Any] | None = None) -> str:
    name = Path(video_path).name.lower()
    if name.startswith("rear") or "rear" in name or (metric or {}).get("view") == "rear":
        return "rear"
    return "side"


def _infer_video_role(video_path: Path, view_type: str | None = None) -> str:
    return infer_video_role_from_name(Path(video_path).name, view_type)


def _read_motionmetrix_values(session_id: str) -> dict[str, Any]:
    path = session_path(session_id) / "motionmetrix_values.json"
    values = read_json(path, {})
    return values if isinstance(values, dict) else {}

def _read_session_meta(session_id: str) -> dict[str, Any]:
    path = session_path(session_id) / "session_meta.json"
    meta = read_json(path, {})
    return meta if isinstance(meta, dict) else {}


def _num_or_none(value):
    if value in (None, "", [], {}):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _analysis_settings(session_id: str, view_type: str) -> dict[str, Any]:
    meta = _read_session_meta(session_id)
    return {
        "progress_direction": meta.get("side_progress_direction", "left_to_right") or "left_to_right",
        "manual_ground_y_px": _num_or_none(meta.get("side_ground_y_px" if view_type == "side" else "rear_ground_y_px")),
        "contact_threshold_px": _num_or_none(meta.get("contact_threshold_px")),
    }


def create_overlay_video(
    session_id: str,
    video_path: Path,
    metric: dict[str, Any],
    start_sec: float = 0.0,
    duration_sec: float = 5.0,
    output_fps: float = 10.0,
    show_all: bool = False,
    progress_callback=None,
    overlay_mode: str = "summary",
) -> dict[str, Any]:
    """Create Skeleton Metric Insight video and modeling CSVs.

    v0.5.5 exports MotionMetrix-definition aligned comparison files and four modeling-oriented files:
    - frame_metrics.csv: frame/time-level skeleton values
    - gait_events.csv: initial-contact/support/contact-time features
    - second_summary.csv: second-level QA summary
    - clip_summary.csv: clip-level feature summary + MotionMetrix target inputs
    """
    if cv2 is None:
        raise RuntimeError("OpenCV가 설치되어 있지 않습니다.")
    if MP_POSE is None:
        raise RuntimeError(dependency_message() or "MediaPipe Pose backend를 사용할 수 없습니다.")

    start_sec = max(0.0, float(start_sec or 0.0))
    duration_sec = max(0.5, min(float(duration_sec or 5.0), 60.0))
    requested_output_fps = float(output_fps or 0.0)
    view_type = _infer_view_type(video_path, metric)
    video_role = _infer_video_role(video_path, view_type)
    settings = _analysis_settings(session_id, view_type)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다.")
    
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        try:
            cap.set(orientation_auto, 1)
        except Exception:
            pass
    
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0    
    
    # v0.5.8: all timing calculations are based on the actual video metadata FPS.
    # The selectable FPS only controls output-preview video size/smoothness.
    if requested_output_fps <= 0:
        output_fps = max(1.0, min(float(source_fps), 60.0))
        output_fps_mode = "auto_from_video_fps"
    else:
        output_fps = max(1.0, min(float(requested_output_fps), 60.0))
        output_fps_mode = "manual_preview_output_fps"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("영상 크기 정보를 읽지 못했습니다.")

    start_frame = int(start_sec * source_fps)
    end_frame = int(min(frame_count if frame_count else start_frame + duration_sec * source_fps, start_frame + duration_sec * source_fps))
    if frame_count and start_frame >= frame_count:
        cap.release()
        raise RuntimeError("시작 시간이 영상 길이를 벗어났습니다.")
    # v0.5.5: analyze every source frame for MotionMetrix comparison stability,
    # while rendering the preview video at the selected output_fps.
    analysis_frame_step = 1
    overlay_frame_step = max(1, int(round(source_fps / output_fps)))
    frame_indices = list(range(start_frame, max(start_frame + 1, end_frame), analysis_frame_step))
    overlay_frame_set = set(range(start_frame, max(start_frame + 1, end_frame), overlay_frame_step))
    expected_frames = max(1, len(frame_indices))

    out_dir = session_path(session_id) / "processed_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_id = metric.get("metric_id", "metric")
    slug = _safe_slug(f"{Path(video_path).stem}_{metric_id}_{stamp}")
    output_video = out_dir / f"{slug}.mp4"
    raw_video = out_dir / f"{slug}_opencv_raw.mp4"
    output_gif = out_dir / f"{slug}_browser_preview.gif"
    frame_metrics_csv = out_dir / f"{slug}_frame_metrics.csv"
    gait_events_csv = out_dir / f"{slug}_gait_events.csv"
    second_summary_csv = out_dir / f"{slug}_second_summary.csv"
    clip_summary_csv = out_dir / f"{slug}_clip_summary.csv"
    legacy_stats_csv = out_dir / f"{slug}_frame_stats.csv"
    # v0.5.7: canonical per-video-role exports so customers can download one CSV
    # with every frame-level Skeleton feature instead of metric-split files.
    canonical_frame_metrics_csv = out_dir / f"{video_role}_all_frame_metrics.csv"
    canonical_gait_events_csv = out_dir / f"{video_role}_gait_events.csv"
    canonical_second_summary_csv = out_dir / f"{video_role}_second_summary.csv"
    canonical_clip_summary_csv = out_dir / f"{video_role}_clip_summary.csv"
    output_meta = out_dir / f"{slug}_meta.json"

    # First pass: analyze every source frame while reusing one MediaPipe Pose backend.
    # static_image_mode=True preserves independent-frame detection semantics.
    frame_items: list[dict[str, Any]] = []
    try:
        with MP_POSE.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose_backend:
            for processed, current_frame in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                points = _detect_landmarks(image, pose_backend=pose_backend)
                timestamp = current_frame / source_fps
                feature = compute_frame_metrics(
                    points,
                    frame_index=current_frame,
                    timestamp_sec=timestamp,
                    source_fps=source_fps,
                    view_type=view_type,
                    progress_direction=settings.get("progress_direction", "left_to_right"),
                )
                frame_items.append(
                    {
                        "frame_index": current_frame,
                        "timestamp_sec": timestamp,
                        "frame_rgb": frame_rgb if current_frame in overlay_frame_set else None,
                        "points": points,
                        "feature": feature,
                    }
                )
                if progress_callback:
                    progress_callback(min(0.45, (processed + 1) / expected_frames * 0.45))
    finally:
        cap.release()

    if not frame_items:
        raise RuntimeError("처리 가능한 프레임이 없습니다. 시작 시간과 영상 길이를 확인하세요.")

    frame_rows = infer_contacts([item["feature"] for item in frame_items], manual_ground_y_px=settings.get("manual_ground_y_px"), contact_threshold_px=settings.get("contact_threshold_px"))
    for item, feature in zip(frame_items, frame_rows):
        item["feature"] = feature
    events = detect_gait_events(frame_rows, view_type=view_type)
    seconds = build_second_summary(frame_rows, events)
    clip_summary = build_clip_summary(frame_rows, events, motionmetrix_values=_read_motionmetrix_values(session_id), session_meta=_read_session_meta(session_id))
    clip_summary.update({"session_id": session_id, "view_type": view_type, "video_role": video_role, "metric_id": metric_id})

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_video), fourcc, output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("결과 영상 파일을 생성하지 못했습니다.")

    h264_width, h264_height = _even_video_size(width, height)
    h264_proc, h264_message = _open_browser_h264_pipe(output_video, h264_width, h264_height, output_fps)
    h264_write_error = ""
    gif_frames: list[Image.Image] = []
    max_gif_frames = 80
    gif_sample_every = max(1, int(math.ceil(len(frame_items) / max_gif_frames)))
    stats_rows: list[dict[str, Any]] = []

    try:
        rendered_count = 0
        for processed, item in enumerate(frame_items):
            points = item["points"]
            feature = item["feature"]
            event = nearest_event(events, float(feature.get("timestamp_sec", 0)))
            stats = compute_reference_values(points, metric.get("metric_id", "")) if points else {"Status": "Pose landmark detection failed"}
            if item.get("frame_rgb") is not None:
                image = Image.fromarray(item["frame_rgb"])
                overlay_img = _draw_pose_overlay_from_points(
                    image,
                    points,
                    metric=metric,
                    show_all=show_all,
                    stats=stats,
                    feature_row=feature,
                    event=event,
                    view_type=view_type,
                    overlay_mode=overlay_mode,
                )
                if rendered_count % gif_sample_every == 0:
                    gif_frames.append(_prepare_gif_frame(overlay_img))
                overlay_rgb = np.array(overlay_img.convert("RGB"))
                overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                writer.write(overlay_bgr)
                rendered_count += 1
                if h264_proc is not None and h264_proc.stdin is not None and not h264_write_error:
                    try:
                        h264_frame = _crop_rgb_to_size(overlay_rgb, h264_width, h264_height)
                        h264_proc.stdin.write(h264_frame.tobytes())
                    except Exception as exc:
                        h264_write_error = f"ffmpeg pipe write failed: {exc}"
                        try:
                            h264_proc.stdin.close()
                        except Exception:
                            pass
                        h264_proc = None
            legacy_row = {
                "frame_index": feature.get("frame_index", ""),
                "timestamp_sec": feature.get("timestamp_sec", ""),
                "metric_id": metric_id,
                "metric_name_kr": metric.get("display_name_kr", ""),
                "metric_name_en": metric.get("display_name_en", ""),
            }
            legacy_row.update({str(k): v for k, v in stats.items()})
            legacy_row.update(feature)
            stats_rows.append(legacy_row)
            if progress_callback:
                progress_callback(0.45 + min(0.45, (processed + 1) / len(frame_items) * 0.45))
    finally:
        writer.release()

    h264_info: dict[str, Any] = {
        "transcoded_for_browser": False,
        "browser_video_codec": "opencv_mp4v_fallback",
        "transcode_message": h264_write_error or h264_message,
        "browser_video_encoder": "none",
    }
    if h264_proc is not None:
        try:
            if h264_proc.stdin is not None:
                h264_proc.stdin.close()
            h264_proc.wait(timeout=120)
            stderr = h264_proc.stderr.read() if h264_proc.stderr is not None else b""
            stderr_msg = stderr.decode("utf-8", errors="ignore") if isinstance(stderr, (bytes, bytearray)) else str(stderr or "")
            if h264_proc.returncode == 0 and output_video.exists() and output_video.stat().st_size > 0:
                h264_info.update({
                    "transcoded_for_browser": True,
                    "browser_video_codec": "h264_yuv420p_avc1",
                    "transcode_message": "Created browser-playable H.264 MP4 directly from RGB frames",
                    "browser_video_encoder": "ffmpeg_rawvideo_pipe",
                    "browser_video_width": h264_width,
                    "browser_video_height": h264_height,
                })
            else:
                h264_info["transcode_message"] = (stderr_msg or h264_info["transcode_message"] or "ffmpeg pipe failed")[:500]
        except Exception as exc:
            h264_info["transcode_message"] = f"ffmpeg pipe finalize failed: {exc}"

    if h264_info.get("transcoded_for_browser"):
        transcode_info = h264_info
    else:
        transcode_info = _transcode_to_browser_mp4(raw_video, output_video)
        if h264_info.get("transcode_message"):
            transcode_info["direct_h264_pipe_message"] = h264_info.get("transcode_message")
    try:
        raw_video.unlink(missing_ok=True)
    except Exception:
        pass
    gif_info = _save_browser_preview_gif(gif_frames, output_gif, output_fps=min(output_fps, 10.0))

    export_run_id = slug
    source_video_name = Path(video_path).name
    for row in frame_rows:
        row.update({"session_id": session_id, "video_role": video_role, "source_video_name": source_video_name, "export_run_id": export_run_id})
    for event in events:
        event.update({"session_id": session_id, "video_role": video_role, "source_video_name": source_video_name, "export_run_id": export_run_id})
    for row in seconds:
        row.update({"session_id": session_id, "video_role": video_role, "source_video_name": source_video_name, "view_type": view_type, "export_run_id": export_run_id})
    for row in stats_rows:
        row.update({"session_id": session_id, "video_role": video_role, "source_video_name": source_video_name, "view_type": view_type, "export_run_id": export_run_id})
    clip_summary.update({"source_video_name": source_video_name, "video_role": video_role, "export_run_id": export_run_id})
    write_csv(frame_metrics_csv, frame_rows)
    write_csv(gait_events_csv, events)
    write_csv(second_summary_csv, seconds)
    write_csv(clip_summary_csv, clip_summary)
    write_csv(legacy_stats_csv, stats_rows)
    # Canonical role-level files overwrite the previous run for the same video slot.
    # They are the recommended files for QA/debugging and final comparison tracing.
    write_csv(canonical_frame_metrics_csv, frame_rows)
    write_csv(canonical_gait_events_csv, events)
    write_csv(canonical_second_summary_csv, seconds)
    write_csv(canonical_clip_summary_csv, clip_summary)

    meta = {
        "created_at": stamp,
        "source_video": str(Path(video_path).relative_to(BASE_DIR)) if Path(video_path).is_relative_to(BASE_DIR) else str(video_path),
        "result_video_path": str(output_video.relative_to(BASE_DIR)),
        "frame_stats_csv_path": str(legacy_stats_csv.relative_to(BASE_DIR)),
        "frame_metrics_csv_path": str(frame_metrics_csv.relative_to(BASE_DIR)),
        "gait_events_csv_path": str(gait_events_csv.relative_to(BASE_DIR)),
        "second_summary_csv_path": str(second_summary_csv.relative_to(BASE_DIR)),
        "clip_summary_csv_path": str(clip_summary_csv.relative_to(BASE_DIR)),
        "all_frame_metrics_csv_path": str(canonical_frame_metrics_csv.relative_to(BASE_DIR)),
        "role_gait_events_csv_path": str(canonical_gait_events_csv.relative_to(BASE_DIR)),
        "role_second_summary_csv_path": str(canonical_second_summary_csv.relative_to(BASE_DIR)),
        "role_clip_summary_csv_path": str(canonical_clip_summary_csv.relative_to(BASE_DIR)),
        "metric_id": metric_id,
        "metric_name_kr": metric.get("display_name_kr", ""),
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "source_fps": round(float(source_fps), 3),
        "analysis_fps": round(float(source_fps), 3),
        "output_fps": output_fps,
        "output_fps_mode": output_fps_mode,
        "low_fps_warning": "60fps 미만: timing/event 기반 지표 정밀 비교 제한" if float(source_fps) < 60 else "",
        "analysis_frame_step": analysis_frame_step,
        "overlay_frame_step": overlay_frame_step,
        "frames_processed": len(frame_rows),
        "overlay_frames_rendered": rendered_count if 'rendered_count' in locals() else 0,
        "gait_events_detected": len(events),
        "show_all_points": show_all,
        "view_type": view_type,
        "video_role": video_role,
        "overlay_mode": overlay_mode,
        "progress_direction": settings.get("progress_direction", "left_to_right"),
        "manual_ground_y_px": settings.get("manual_ground_y_px"),
        "contact_threshold_px": settings.get("contact_threshold_px"),
    }
    meta.update(transcode_info)
    meta.update(gif_info)
    write_json(output_meta, meta)
    index_path = session_path(session_id) / "processed_videos.json"
    index = read_json(index_path, [])
    if not isinstance(index, list):
        index = []
    index.append(meta)
    write_json(index_path, index)
    return meta


def inspect_frame_features(session_id: str, video_path: Path, frame_index: int | None = None, timestamp_sec: float | None = None, metric: dict[str, Any] | None = None, show_all: bool = True) -> dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("OpenCV가 설치되어 있지 않습니다.")
    if MP_POSE is None:
        raise RuntimeError(dependency_message() or "MediaPipe Pose backend를 사용할 수 없습니다.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다.")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_index is None:
        frame_index = int(max(0.0, float(timestamp_sec or 0.0)) * source_fps)
    frame_index = max(0, int(frame_index))
    if total_frames:
        frame_index = min(frame_index, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("해당 프레임을 읽지 못했습니다.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    points = _detect_landmarks(image)
    view_type = _infer_view_type(video_path, metric or {})
    settings = _analysis_settings(session_id, view_type)
    row = compute_frame_metrics(points, frame_index=frame_index, timestamp_sec=frame_index / source_fps, source_fps=source_fps, view_type=view_type, progress_direction=settings.get("progress_direction", "left_to_right"))
    row = infer_contacts([row], manual_ground_y_px=settings.get("manual_ground_y_px"), contact_threshold_px=settings.get("contact_threshold_px"))[0]
    overlay = _draw_pose_overlay_from_points(image, points, metric=metric or {}, show_all=show_all, stats={}, feature_row=row, event=None, view_type=view_type, overlay_mode="summary")
    return {"image": overlay, "frame_features": row, "source_fps": source_fps, "total_frames": total_frames, "view_type": view_type}

def processed_video_results(session_id: str) -> list[dict[str, Any]]:
    rows = read_json(session_path(session_id) / "processed_videos.json", [])
    return rows if isinstance(rows, list) else []


def save_preview_result(session_id: str, image: Image.Image, meta: dict[str, Any]) -> str:
    base = session_path(session_id) / "preview"
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = base / f"preview_{stamp}.png"
    image.save(image_path)
    result = {
        "created_at": stamp,
        "image_path": str(image_path.relative_to(BASE_DIR)),
        **meta,
    }
    path = session_path(session_id) / "preview_results.json"
    rows = read_json(path, [])
    if not isinstance(rows, list):
        rows = []
    rows.append(result)
    write_json(path, rows)
    return result["image_path"]


def preview_summary_for_session(session_id: str) -> dict[str, Any]:
    rows = read_json(session_path(session_id) / "preview_results.json", [])
    if not isinstance(rows, list) or not rows:
        return {
            "preview_count": 0,
            "latest_preview_image_path": "",
            "latest_preview_metric_id": "",
            "latest_preview_step_id": "",
            "latest_preview_stats": "",
        }
    latest = rows[-1]
    return {
        "preview_count": len(rows),
        "latest_preview_image_path": latest.get("image_path", ""),
        "latest_preview_metric_id": latest.get("metric_id", ""),
        "latest_preview_step_id": latest.get("step_id", ""),
        "latest_preview_stats": json.dumps(latest.get("stats", {}), ensure_ascii=False),
    }
