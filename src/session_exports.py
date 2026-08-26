from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .storage import BASE_DIR, EXPORTS_DIR, read_json, session_path


def infer_video_role_from_name(name: str | None, view_type: str | None = None) -> str:
    text = str(name or "").lower()
    view = str(view_type or "").lower()
    if "rear_static" in text:
        return "rear_static"
    if "rear_running" in text or ("rear" in text and "running" in text):
        return "rear_running"
    if "side_static" in text:
        return "side_static"
    if "side_running" in text or ("side" in text and "running" in text):
        return "side_running"
    if text.startswith("rear") or view == "rear":
        return "rear_running" if "running" in text else "rear_unknown"
    if text.startswith("side") or view == "side":
        return "side_running" if "running" in text else "side_unknown"
    return "unknown"


def _csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except Exception:
        return str(path)


def processed_csv_index(session_id: str) -> list[dict[str, Any]]:
    base = session_path(session_id)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_from_meta(meta: dict[str, Any]) -> None:
        source_video = meta.get("source_video") or meta.get("source_video_name") or ""
        view_type = meta.get("view_type", "")
        role = meta.get("video_role") or infer_video_role_from_name(source_video, view_type)
        for kind, key in [
            ("frame_metrics", "frame_metrics_csv_path"),
            ("frame_stats", "frame_stats_csv_path"),
            ("gait_events", "gait_events_csv_path"),
            ("second_summary", "second_summary_csv_path"),
            ("clip_summary", "clip_summary_csv_path"),
        ]:
            rel = meta.get(key)
            if not rel:
                continue
            path = BASE_DIR / rel
            if not path.exists():
                continue
            token = str(path.resolve())
            if token in seen:
                continue
            seen.add(token)
            out.append({
                "session_id": session_id,
                "kind": kind,
                "video_role": role,
                "view_type": view_type,
                "source_video": source_video,
                "metric_id": meta.get("metric_id", ""),
                "created_at": meta.get("created_at", ""),
                "path": path,
                "relative_path": _rel(path),
                "mtime": path.stat().st_mtime,
            })

    rows = read_json(base / "processed_videos.json", [])
    if isinstance(rows, list):
        for meta in rows:
            if isinstance(meta, dict):
                add_from_meta(meta)

    pdir = base / "processed_videos"
    if pdir.exists():
        for path in pdir.glob("*.csv"):
            token = str(path.resolve())
            if token in seen:
                continue
            stem = path.stem.lower()
            if stem.endswith("_frame_metrics"):
                kind = "frame_metrics"
            elif stem.endswith("_frame_stats"):
                kind = "frame_stats"
            elif stem.endswith("_gait_events"):
                kind = "gait_events"
            elif stem.endswith("_second_summary"):
                kind = "second_summary"
            elif stem.endswith("_clip_summary"):
                kind = "clip_summary"
            else:
                continue
            role = infer_video_role_from_name(path.name)
            view = "rear" if role.startswith("rear") else "side" if role.startswith("side") else "unknown"
            seen.add(token)
            out.append({
                "session_id": session_id,
                "kind": kind,
                "video_role": role,
                "view_type": view,
                "source_video": "",
                "metric_id": "",
                "created_at": "",
                "path": path,
                "relative_path": _rel(path),
                "mtime": path.stat().st_mtime,
            })
    return out


def _latest_by_role(items: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if item.get("kind") != kind:
            continue
        role = item.get("video_role") or "unknown"
        if role.endswith("_unknown") or role == "unknown":
            continue
        if role not in result or float(item.get("mtime", 0)) >= float(result[role].get("mtime", 0)):
            result[role] = item
    return result


def latest_csv_by_role(session_id: str, kind: str) -> dict[str, dict[str, Any]]:
    return _latest_by_role(processed_csv_index(session_id), kind)


def create_session_all_skeleton_frames(session_id: str) -> Path | None:
    items = latest_csv_by_role(session_id, "frame_metrics")
    frames = []
    for role, item in sorted(items.items()):
        df = _csv(item["path"])
        if df.empty:
            continue
        if "video_role" in df.columns:
            df["video_role"] = role
        else:
            df.insert(0, "video_role", role)
        if "source_frame_csv" in df.columns:
            df["source_frame_csv"] = item["relative_path"]
        else:
            df.insert(1, "source_frame_csv", item["relative_path"])
        frames.append(df)
    path = EXPORTS_DIR / f"{session_id}_session_all_skeleton_frames.csv"
    if not frames:
        return None
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def create_session_gait_events(session_id: str) -> Path | None:
    items = latest_csv_by_role(session_id, "gait_events")
    frames = []
    for role, item in sorted(items.items()):
        df = _csv(item["path"])
        if df.empty:
            continue
        if "video_role" in df.columns:
            df["video_role"] = role
        else:
            df.insert(0, "video_role", role)
        if "source_event_csv" in df.columns:
            df["source_event_csv"] = item["relative_path"]
        else:
            df.insert(1, "source_event_csv", item["relative_path"])
        frames.append(df)
    path = EXPORTS_DIR / f"{session_id}_session_gait_events.csv"
    if not frames:
        return None
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def create_session_skeleton_metric_summary(session_id: str, final_rows: list[dict[str, Any]] | None = None) -> Path:
    if final_rows is None:
        from .comparison import build_final_comparison_rows
        final_rows = build_final_comparison_rows(session_id)
    path = EXPORTS_DIR / f"{session_id}_session_skeleton_metric_summary.csv"
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final_rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def create_debug_export_package(session_id: str, extra_paths: list[Path] | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = EXPORTS_DIR / f"{session_id}_debug_export_package_{stamp}.zip"
    extra_paths = extra_paths or []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in extra_paths:
            if p and p.exists() and p.is_file():
                zf.write(p, p.name)
        for item in processed_csv_index(session_id):
            p = item["path"]
            if p.exists() and p.is_file():
                arc = f"processed/{item.get('video_role','unknown')}/{p.name}"
                zf.write(p, arc)
        for name in ["session_meta.json", "motionmetrix_values.json", "visual_labels.json", "videos.json", "processed_videos.json"]:
            p = session_path(session_id) / name
            if p.exists():
                zf.write(p, f"session/{name}")
    return zip_path


def export_session_debug_files(session_id: str, final_rows: list[dict[str, Any]] | None = None) -> list[Path]:
    paths: list[Path] = []
    for p in [
        create_session_all_skeleton_frames(session_id),
        create_session_gait_events(session_id),
        create_session_skeleton_metric_summary(session_id, final_rows=final_rows),
    ]:
        if p and p.exists():
            paths.append(p)
    paths.append(create_debug_export_package(session_id, paths))
    return paths
