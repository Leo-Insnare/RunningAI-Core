import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
EXPORTS_DIR = BASE_DIR / "exports"
DEFINITION_DIR = DATA_DIR / "metric_definitions"

for path in [SESSIONS_DIR, EXPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def new_session_id():
    return f"SESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def session_path(session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "_-.")
    return SESSIONS_DIR / safe


def list_sessions():
    if not SESSIONS_DIR.exists():
        return []
    return sorted([p.name for p in SESSIONS_DIR.iterdir() if p.is_dir()], reverse=True)


def read_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(session_id: str):
    base = session_path(session_id)
    return {
        "session_meta": read_json(base / "session_meta.json", {}),
        "motionmetrix_values": read_json(base / "motionmetrix_values.json", {}),
        "visual_labels": read_json(base / "visual_labels.json", {}),
        "review_status": read_json(base / "review_status.json", {}),
        "videos": read_json(base / "videos.json", {}),
    }


def save_session(session_id: str, session_meta: dict, values: dict, visual_labels: dict, review_status: dict):
    base = session_path(session_id)
    base.mkdir(parents=True, exist_ok=True)
    write_json(base / "session_meta.json", session_meta)
    write_json(base / "motionmetrix_values.json", values)
    write_json(base / "visual_labels.json", visual_labels)
    write_json(base / "review_status.json", review_status)
    definition_file = None
    for name in ["metric_definitions_v0_5_12.json", "metric_definitions_v0_5_11.json", "metric_definitions_v0_5_10.json", "metric_definitions_v0_5_9.json", "metric_definitions_v0_5_8.json", "metric_definitions_v0_5_7.json", "metric_definitions_v0_5_6.json", "metric_definitions_v0_5_5.json", "metric_definitions_v0_5_4.json", "metric_definitions_v0_5_3.json", "metric_definitions_v0_4_8.json", "metric_definitions_v0_4_5.json", "metric_definitions_v0_4.json", "metric_definitions_v0_3.json", "metric_definitions_v0_2.json", "metric_definitions_v0_1.json"]:
        candidate = DEFINITION_DIR / name
        if candidate.exists():
            definition_file = candidate
            break
    snapshot = read_json(definition_file, {}) if definition_file else {}
    write_json(base / "metric_definitions_snapshot.json", snapshot)


def video_paths_for_session(session_id: str):
    data = read_json(session_path(session_id) / "videos.json", {})
    return data if isinstance(data, dict) else {}


def save_uploaded_file(session_id: str, uploaded_file, filename: str, slot: str | None = None):
    base = session_path(session_id) / "videos"
    base.mkdir(parents=True, exist_ok=True)
    target = base / filename
    target.write_bytes(uploaded_file.getbuffer())
    relative_path = str(target.relative_to(BASE_DIR))
    if slot:
        videos = video_paths_for_session(session_id)
        videos[f"{slot}_video_path"] = relative_path
        videos[f"{slot}_original_filename"] = uploaded_file.name
        videos[f"{slot}_size_mb"] = round(target.stat().st_size / 1024 / 1024, 3)
        write_json(session_path(session_id) / "videos.json", videos)
    return relative_path


def existing_videos(session_id: str):
    base = session_path(session_id) / "videos"
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_file()])


def make_backup_zip():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = EXPORTS_DIR / f"labeling_data_backup_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in [SESSIONS_DIR, EXPORTS_DIR, DEFINITION_DIR]:
            if not root.exists():
                continue
            for file in root.rglob("*"):
                if file.is_file() and file != zip_path:
                    zf.write(file, file.relative_to(BASE_DIR))
    return zip_path


def install_sample_session():
    sample_dir = BASE_DIR / "sample_data" / "sample_session"
    if not sample_dir.exists():
        return None
    target = SESSIONS_DIR / "SAMPLE_001"
    if target.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = SESSIONS_DIR / f"SAMPLE_001_{stamp}"
    shutil.copytree(sample_dir, target)
    return target.name
