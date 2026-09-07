from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gait_features import is_low_fps
from src.pose_preview import (
    _parse_ffmpeg_video_info,
    _prepare_analysis_video,
    _validate_pose_orientation,
    probe_video_info,
)


def test_rotation_parser_uses_display_matrix_direction():
    text = "displaymatrix: rotation of -90.00 degrees\nVideo: hevc, yuv420p, 1920x1080, 60 fps"
    rotation, width, height = _parse_ffmpeg_video_info(text)
    assert rotation == 90
    assert (width, height) == (1920, 1080)


def test_nominal_60fps_is_not_low_fps():
    assert not is_low_fps(59.775)
    assert not is_low_fps(59.94)
    assert is_low_fps(30.0)


def test_rotated_mp4_is_physically_normalized(tmp_path: Path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    base = tmp_path / "base.mp4"
    rotated = tmp_path / "rotated.mp4"

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x96:r=30:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(base),
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-display_rotation:v:0",
            "-90",
            "-i",
            str(base),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )

    info = probe_video_info(rotated)
    assert info["rotation_clockwise_deg"] == 90
    assert (info["width"], info["height"]) == (96, 160)

    normalized, orientation = _prepare_analysis_video(rotated, info)
    normalized_info = probe_video_info(normalized)

    assert orientation["orientation_normalized"] is True
    assert normalized != rotated
    assert normalized_info["rotation_clockwise_deg"] == 0
    assert (normalized_info["width"], normalized_info["height"]) == (96, 160)


def _pose_frame(shoulder_x: float, shoulder_y: float, pelvis_x: float, pelvis_y: float):
    return {
        "points": {
            11: {"x": shoulder_x - 2, "y": shoulder_y, "visibility": 1.0},
            12: {"x": shoulder_x + 2, "y": shoulder_y, "visibility": 1.0},
            23: {"x": pelvis_x - 2, "y": pelvis_y, "visibility": 1.0},
            24: {"x": pelvis_x + 2, "y": pelvis_y, "visibility": 1.0},
        }
    }


def test_pose_orientation_sanity_accepts_upright_torso():
    frames = [_pose_frame(80, 30, 80, 100) for _ in range(12)]
    result = _validate_pose_orientation(frames)
    assert result["checked"] is True
    assert result["median_torso_angle_deg"] == 0.0


def test_pose_orientation_sanity_rejects_sideways_torso():
    frames = [_pose_frame(120, 60, 30, 60) for _ in range(12)]
    try:
        _validate_pose_orientation(frames)
    except RuntimeError as exc:
        assert "영상 방향" in str(exc)
    else:
        raise AssertionError("sideways pose must be rejected")
