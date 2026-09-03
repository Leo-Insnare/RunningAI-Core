from pathlib import Path
import base64
import hashlib
import subprocess

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.definitions import (
    all_metric_fields,
    all_session_fields,
    all_visual_fields,
    derived_lookup,
    keypoint_lookup,
    load_definitions,
    metric_categories,
)
from src.exporter import export_all, missing_values_for_session
from src.comparison import build_final_comparison_rows, compute_auto_motionmetrix_values
from src.storage import (
    BASE_DIR,
    EXPORTS_DIR,
    existing_videos,
    list_sessions,
    load_session,
    make_backup_zip,
    install_sample_session,
    new_session_id,
    save_session,
    save_uploaded_file,
    session_path,
    video_paths_for_session,
)
from src.ui_components import (
    option_label,
    render_direct_input_notice,
    render_direct_input_style,
    render_input,
    render_metric_guide,
    render_skeleton_graph,
    render_skeleton_output_notice,
    widget_key,
)
from src.api_integration import (
    analysis_summary_rows,
    analyze_session,
    inference_status,
    load_saved_analysis,
    resolve_inference_settings,
    save_analysis_result,
    sync_analysis_sheet,
    validate_result_export_parity,
)
from src.pose_preview import (
    WORKFLOW_STEPS,
    dependencies_status,
    dependency_message,
    create_overlay_video,
    extract_frame,
    overlay_pose,
    inspect_frame_features,
    processed_video_results,
    save_preview_result,
)

st.set_page_config(page_title="RunningAI · 달리기 자세 AI 분석", page_icon="🏃", layout="wide")
render_direct_input_style()

metrics_defs, keypoint_defs, visual_defs = load_definitions()
keypoints = keypoint_lookup(keypoint_defs)
derived = derived_lookup(keypoint_defs)

CATEGORY_LABELS = {
    "side_biomechanics": "측면 생체역학 지표",
    "side_joint_rom": "측면 관절 운동범위",
    "side_manual_or_derived": "측면 수동/추후 계산 지표",
    "side_manual_label": "측면 수동 라벨",
    "temporal_gait": "시간 기반 보행/러닝 지표",
    "rear_biomechanics": "후면 생체역학 지표",
    "aggregate_motionmetrix": "종합 MotionMetrix 지표",
    "optional_motionmetrix": "선택 입력 지표",
    "side_derived_later": "3차 계산 후보 지표",
}

VIDEO_SLOTS = {
    "side_static": "측면 전신 정지 영상 또는 정지 구간",
    "side_running": "측면 달리기 영상",
    "rear_static": "후면 전신 정지 영상 또는 정지 구간",
    "rear_running": "후면 달리기 영상",
}

def _file_digest(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _preview_video_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False

    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return False
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            ok, frame = cap.read()
            return (
                width > 0
                and height > 0
                and frames > 0
                and ok
                and frame is not None
            )
        finally:
            cap.release()
    except Exception:
        return False


def _orientation_safe_preview(session_id: str, slot: str, source):
    cache_dir = session_path(session_id) / ".video_preview"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(source, Path):
        source_path = source
        digest = _file_digest(source_path)
    else:
        buffer = source.getbuffer()
        digest = hashlib.sha1(buffer).hexdigest()[:12]
        suffix = Path(source.name).suffix.lower() or ".mp4"
        source_path = cache_dir / f"{slot}_{digest}_source{suffix}"
        if not source_path.exists():
            source_path.write_bytes(buffer)

    normalized_path = cache_dir / f"{slot}_{digest}_preview_v2.mp4"
    temp_path = cache_dir / f"{slot}_{digest}_preview_v2.tmp.mp4"

    if _preview_video_valid(normalized_path):
        return normalized_path

    try:
        normalized_path.unlink(missing_ok=True)
        temp_path.unlink(missing_ok=True)
    except Exception:
        pass

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-map_metadata",
            "-1",
            "-an",
            "-vf",
            "scale=w=min(720\\,iw):h=-2,fps=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "avc1",
            "-movflags",
            "+faststart",
            str(temp_path),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=240,
        )
        if proc.returncode == 0 and _preview_video_valid(temp_path):
            temp_path.replace(normalized_path)
            return normalized_path
    except Exception:
        pass
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return source_path


def init_state():
    st.session_state.setdefault("current_session_id", "")
    st.session_state.setdefault("loaded_once", False)
    st.session_state.setdefault("show_required_only", False)


def set_widget_value(field_id, value):
    st.session_state[widget_key(field_id)] = value

def load_to_widgets(session_id):
    data = load_session(session_id)
    for field in all_session_fields(metrics_defs):
        if field["field_id"] in data["session_meta"]:
            set_widget_value(field["field_id"], data["session_meta"][field["field_id"]])
    for field in all_metric_fields(metrics_defs):
        if field["field_id"] in data["motionmetrix_values"]:
            set_widget_value(field["field_id"], data["motionmetrix_values"][field["field_id"]])
    for field in all_visual_fields(visual_defs):
        if field["field_id"] in data["visual_labels"]:
            set_widget_value(field["field_id"], data["visual_labels"][field["field_id"]])


def collect_values(fields):
    values = {}
    for field in fields:
        key = widget_key(field["field_id"])
        value = st.session_state.get(key, "")
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        values[field["field_id"]] = value
    return values


def collect_session():
    meta = collect_values(all_session_fields(metrics_defs))
    values = collect_values(all_metric_fields(metrics_defs))
    visual = collect_values(all_visual_fields(visual_defs))
    session_id = st.session_state.get("current_session_id") or meta.get("session_id")
    if session_id:
        meta.update(video_paths_for_session(session_id))
    return meta, values, visual


def save_current_session():
    session_id = st.session_state.get("current_session_id") or st.session_state.get(widget_key("session_id"))
    if not session_id:
        session_id = new_session_id()
    st.session_state["current_session_id"] = session_id
    meta, values, visual = collect_session()
    pending_metric_updates = st.session_state.get("pending_metric_updates", {})
    if pending_metric_updates:
        values.update(pending_metric_updates)
        st.session_state["pending_metric_updates"] = {}
    # v0.5.3: automatically fill derived MotionMetrix values requested by customer
    # (Hip/Knee ROM, overall averages, left-right differences, unit-normalized averages).
    values.update(compute_auto_motionmetrix_values(values))
    meta["session_id"] = session_id
    missing = missing_values_preview(session_id, meta, values, visual)
    review_status = {
        "missing_core_count": len(missing),
        "is_ready_for_export": len(missing) == 0,
    }
    save_session(session_id, meta, values, visual, review_status)
    return session_id, review_status


def missing_values_preview(session_id, meta, values, visual):
    missing = []
    for field in all_session_fields(metrics_defs):
        if field.get("required") and not meta.get(field["field_id"]):
            missing.append({"section": "세션 정보", "field_id": field["field_id"], "label_kr": field["label_kr"]})
    for metric in metrics_defs["metrics"]:
        if metric.get("required_level") != "core":
            continue
        for field in metric.get("fields", []):
            if field.get("required") and not values.get(field["field_id"]):
                missing.append({"section": metric["display_name_kr"], "field_id": field["field_id"], "label_kr": field["label_kr"]})
    for field in all_visual_fields(visual_defs):
        if field.get("required") and not visual.get(field["field_id"]):
            missing.append({"section": "육안 자세 평가", "field_id": field["field_id"], "label_kr": field["label_kr"]})
    return missing


def completion_summary(meta, values, visual):
    session_required = [f for f in all_session_fields(metrics_defs) if f.get("required")]
    metric_required = []
    for metric in metrics_defs["metrics"]:
        if metric.get("required_level") == "core":
            metric_required.extend(metric.get("fields", []))
    visual_required = [f for f in all_visual_fields(visual_defs) if f.get("required")]
    total = len(session_required) + len(metric_required) + len(visual_required)
    done = 0
    done += sum(1 for f in session_required if meta.get(f["field_id"]))
    done += sum(1 for f in metric_required if values.get(f["field_id"]))
    done += sum(1 for f in visual_required if visual.get(f["field_id"]))
    return done, total


def metric_filter(categories):
    return [m for m in metrics_defs["metrics"] if m["category"] in categories]


def render_metrics(metrics):
    required_only = st.session_state.get("show_required_only", False)
    if required_only:
        metrics = [m for m in metrics if m.get("required_level") == "core"]
    grouped = {}
    for metric in metrics:
        grouped.setdefault(metric["category"], []).append(metric)
    for category, category_metrics in grouped.items():
        st.markdown(f"### {CATEGORY_LABELS.get(category, category)}")
        for metric in category_metrics:
            title = f"{metric['display_name_kr']} · {metric['display_name_en']}"
            with st.expander(title, expanded=metric.get("required_level") == "core"):
                st.caption(f"분류: {CATEGORY_LABELS.get(metric['category'], metric['category'])} / 입력 출처: {metric['source_type']} / 3차 역할: {metric['model_role']}")
                render_metric_guide(metric, keypoints, derived)
                if any(f.get("required") for f in metric.get("fields", [])):
                    render_direct_input_notice("🟧 아래 주황색 표시 항목은 고객이 MotionMetrix 결과값을 직접 입력해야 하는 핵심 영역입니다.")
                cols = st.columns(2)
                fields = metric.get("fields", [])
                if st.session_state.get("show_required_only", False):
                    fields = [f for f in fields if f.get("required")]
                if not fields:
                    st.info("필수값만 보기 모드에서 표시할 필수 입력 필드가 없습니다.")
                for idx, field in enumerate(fields):
                    with cols[idx % 2]:
                        render_input(field, st.session_state.get(widget_key(field["field_id"])))
                _render_auto_calculation_panel(metric["metric_id"])




def _current_motionmetrix_values_from_widgets() -> dict:
    values = {}
    for field in all_metric_fields(metrics_defs):
        key = widget_key(field["field_id"])
        if key in st.session_state:
            values[field["field_id"]] = st.session_state.get(key)
    return compute_auto_motionmetrix_values(values)


def _render_auto_calculation_panel(metric_id: str):
    values = _current_motionmetrix_values_from_widgets()
    if metric_id == "hip_thigh_flexion_extension":
        rows = [
            {"자동 계산 항목": "왼쪽 고관절 ROM", "값": values.get("left_hip_rom_deg", ""), "단위": "deg"},
            {"자동 계산 항목": "오른쪽 고관절 ROM", "값": values.get("right_hip_rom_deg", ""), "단위": "deg"},
            {"자동 계산 항목": "고관절 ROM 전체 평균", "값": values.get("hip_rom_mean_deg", ""), "단위": "deg"},
            {"자동 계산 항목": "고관절 ROM 좌우 차이", "값": values.get("hip_rom_asymmetry_deg", ""), "단위": "deg"},
        ]
    elif metric_id == "knee_flexion_rom":
        rows = [
            {"자동 계산 항목": "왼쪽 슬관절 ROM", "값": values.get("left_knee_rom_deg", ""), "단위": "deg"},
            {"자동 계산 항목": "오른쪽 슬관절 ROM", "값": values.get("right_knee_rom_deg", ""), "단위": "deg"},
            {"자동 계산 항목": "슬관절 ROM 전체 평균", "값": values.get("knee_rom_mean_deg", ""), "단위": "deg"},
            {"자동 계산 항목": "슬관절 ROM 좌우 차이", "값": values.get("knee_rom_asymmetry_deg", ""), "단위": "deg"},
        ]
    else:
        return
    st.markdown("#### 자동 계산값")
    st.caption("고객 요청에 따라 직접 입력하지 않고 저장 시 자동 계산되는 값입니다.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_final_comparison_table(session_id: str):
    rows = build_final_comparison_rows(session_id) if session_id else []
    if not rows:
        st.info("최종 비교표를 생성할 세션 데이터가 없습니다.")
        return
    df = pd.DataFrame(rows)
    st.markdown("#### 최종 비교표: Skeleton adjusted value / MotionMetrix value")
    st.caption("고객 확인이 쉽도록 앞 열에 최종 Skeleton adjusted/selected 값과 MotionMetrix 값을 나란히 배치했습니다. raw/audit, source, FPS, scale confidence 등 부수 지표는 뒤 열에서 확인할 수 있습니다.")
    def _style(row):
        status = str(row.get("비교 상태", ""))
        styles = [""] * len(row)
        for i, col in enumerate(row.index):
            if "Skeleton" in col:
                styles[i] = "background-color: #dbeafe"
            if "MotionMetrix" in col:
                styles[i] = "background-color: #ffedd5"
            if col == "비교 상태":
                if "비교 가능" in status:
                    styles[i] = "background-color: #dcfce7"
                elif "입력" in status or "단위" in status or "없음" in status:
                    styles[i] = "background-color: #fef3c7"
        return styles
    st.dataframe(df.style.apply(_style, axis=1), hide_index=True, use_container_width=True)

def _start_new_analysis_session() -> str:
    sid = new_session_id()
    st.session_state["current_session_id"] = sid
    set_widget_value("session_id", sid)
    return sid


def render_sidebar():
    with st.sidebar:
        st.title("🏃 RunningAI")
        st.caption("달리기 영상 기반 자세 분석 · v0.16 Frozen")

        if st.button("새 분석 시작", type="primary", use_container_width=True):
            sid = _start_new_analysis_session()
            st.success(f"새 분석 ID: {sid}")

        sessions = list_sessions()
        selected = st.selectbox("이전 분석 불러오기", [""] + sessions)
        if st.button("선택 분석 불러오기", use_container_width=True, disabled=not selected):
            st.session_state["current_session_id"] = selected
            load_to_widgets(selected)
            st.success(f"불러옴: {selected}")

        if st.button("현재 분석 정보 저장", use_container_width=True):
            sid, _ = save_current_session()
            st.success(f"저장 완료: {sid}")

        st.divider()
        if st.button("샘플 분석 불러오기", use_container_width=True):
            sid = install_sample_session()
            if sid:
                st.session_state["current_session_id"] = sid
                load_to_widgets(sid)
                st.success(f"샘플 분석 불러오기 완료: {sid}")
            else:
                st.warning("sample_data/sample_session 폴더를 찾을 수 없습니다.")

        st.divider()
        st.caption("현재 분석 ID")
        st.code(st.session_state.get("current_session_id") or "새 분석을 시작하세요")
        st.caption("본 화면은 최종 AI 분석 결과 확인용입니다. 자세한 Skeleton/Debug 정보는 결과 화면의 고급 정보에서 확인할 수 있습니다.")


def tab_session_info():
    st.subheader("1. 세션 정보")
    st.write("검사 1건을 하나의 세션으로 저장합니다. 이름 대신 사용자 ID 사용을 권장합니다.")
    cols = st.columns(2)
    for idx, field in enumerate(all_session_fields(metrics_defs)):
        with cols[idx % 2]:
            render_input(field, st.session_state.get(widget_key(field["field_id"])))
    if st.button("세션 정보 저장", type="primary"):
        sid, _ = save_current_session()
        st.success(f"저장 완료: {sid}")



def _read_csv_preview(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _render_processed_feature_tables(meta: dict, prefix: str = "latest"):
    """Show generated skeleton feature datasets in-app so customers do not need Excel first."""
    csv_items = [
        ("통합 프레임 Raw", "all_frame_metrics_csv_path", "v0.5.12 권장: 해당 후방/측방 영상의 모든 프레임별 Skeleton raw·좌표·각도·이벤트 상태를 한 파일에서 확인합니다."),
        ("프레임별 Skeleton Feature", "frame_metrics_csv_path", "현재 처리 구간의 프레임/시점별 좌표·각도·거리·접촉 상태입니다."),
        ("착지 이벤트 Feature", "gait_events_csv_path", "착지 시점, 초기 지지 구간, 접촉시간, 착지 무릎/정강이/발 각도, 골반-발목 거리입니다."),
        ("초별 요약", "second_summary_csv_path", "고객 검수용 초 단위 평균/이벤트 수 요약입니다."),
        ("클립 요약 + MotionMetrix Target", "clip_summary_csv_path", "3차 모델링용 영상 단위 feature 요약과 MotionMetrix 직접 입력값 연결입니다."),
    ]
    st.markdown("#### 화면에서 바로 확인하는 Skeleton Feature 표")
    tabs = st.tabs([item[0] for item in csv_items])
    for tab, (title, key_name, desc) in zip(tabs, csv_items):
        with tab:
            path = BASE_DIR / meta.get(key_name, "") if meta.get(key_name) else None
            st.caption(desc)
            if path and path.exists():
                df = _read_csv_preview(path)
                if not df.empty:
                    st.dataframe(df.head(200), hide_index=True, use_container_width=True)
                else:
                    st.info("표시할 데이터가 없습니다.")
                _download_file_button(path, f"{title} CSV 다운로드", "text/csv", key=f"download::{prefix}::{key_name}::{path.name}")
            else:
                st.info("해당 CSV가 아직 생성되지 않았습니다.")


def _render_frame_inspector(session_id: str, video_options: dict[str, Path], metric_candidates: list[dict], selected_video_name: str):
    st.markdown("### 4) 프레임 상세 확인 / Skeleton Feature Inspector")
    st.caption("프레임 번호 또는 초를 입력하면 해당 시점의 skeleton 계산값을 화면에서 바로 확인할 수 있습니다. 엑셀을 열지 않아도 현재 프레임의 feature와 가장 가까운 착지 이벤트 값을 검수할 수 있습니다.")
    metric_labels = {f"{m['display_name_kr']} · {m['metric_id']}": m for m in metric_candidates}
    c0, c1, c2, c3 = st.columns([1.5, 1, 1, 1])
    with c0:
        inspect_video_name = st.selectbox("상세 확인 영상", list(video_options.keys()), index=list(video_options.keys()).index(selected_video_name) if selected_video_name in video_options else 0, key="inspect_video_source")
    with c1:
        inspect_by = st.radio("선택 방식", ["프레임", "초"], horizontal=True, key="inspect_by")
    with c2:
        frame_number = st.number_input("프레임 번호", min_value=0, value=0, step=1, key="inspect_frame_number", disabled=inspect_by != "프레임")
    with c3:
        inspect_sec = st.number_input("시점(sec)", min_value=0.0, value=0.0, step=0.1, key="inspect_sec", disabled=inspect_by != "초")
    inspect_metric = metric_labels[st.selectbox("상세 기준 지표", list(metric_labels.keys()), key="inspect_metric")]
    if st.button("선택 프레임 Feature 확인", use_container_width=True):
        try:
            result = inspect_frame_features(
                session_id=session_id,
                video_path=video_options[inspect_video_name],
                frame_index=int(frame_number) if inspect_by == "프레임" else None,
                timestamp_sec=float(inspect_sec) if inspect_by == "초" else None,
                metric=inspect_metric,
                show_all=True,
            )
            st.image(result["image"], caption="선택 프레임 Skeleton Metric Overlay", use_container_width=True)
            feature_df = pd.DataFrame([result["frame_features"]])
            st.markdown("#### 선택 프레임 Skeleton 계산값")
            st.dataframe(feature_df.T.rename(columns={0: "value"}), use_container_width=True)
            st.caption("이 값은 MotionMetrix 정답값이 아니라, 3차 모델링에서 X feature로 사용할 skeleton 기반 참고값입니다.")
        except Exception as exc:
            st.error(f"프레임 상세 확인 실패: {exc}")

def _default_analysis_window(session_id: str, video_name: str) -> tuple[float, float]:
    """Return MotionMetrix-style default analysis window from session meta.

    The UI remains the same as v0.5.4, but v0.5.5 defaults to the stored
    side/rear measurement interval so final Skeleton averages are not computed
    only from the first preview seconds by mistake.
    """
    try:
        meta = load_session(session_id).get("session_meta", {}) if session_id else {}
    except Exception:
        meta = {}
    name = (video_name or "").lower()
    prefix = "rear" if name.startswith("rear") or "rear" in name else "side"
    start = meta.get(f"{prefix}_measure_start_sec", 0.0) or 0.0
    end = meta.get(f"{prefix}_measure_end_sec", None)
    try:
        start_f = max(0.0, float(start))
    except Exception:
        start_f = 0.0
    try:
        end_f = float(end) if end not in (None, "") else start_f + 5.0
    except Exception:
        end_f = start_f + 5.0
    duration = max(0.5, min(60.0, end_f - start_f if end_f > start_f else 5.0))
    return start_f, duration


def tab_videos():
    st.subheader("3. 영상 업로드 / Skeleton 결과 영상")
    st.write("측면/후면 영상을 세션 폴더에 저장하고, 업로드한 영상에서 Skeleton Overlay 결과 영상을 생성할 수 있습니다.")
    session_id = st.session_state.get("current_session_id") or st.session_state.get(widget_key("session_id"))
    if not session_id:
        st.warning("먼저 새 세션을 생성하거나 세션 ID를 입력하세요.")
        return

    st.markdown("### 1) 원본 영상 업로드")
    for slot, label in VIDEO_SLOTS.items():
        st.markdown(f"#### {label}")
        uploaded = st.file_uploader(label, type=["mp4", "mov", "avi", "m4v"], key=f"upload::{slot}")
        if uploaded:
            filename = f"{slot}{Path(uploaded.name).suffix.lower()}"
            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button(f"{label} 저장", key=f"save_upload::{slot}"):
                    rel = save_uploaded_file(session_id, uploaded, filename, slot=slot)
                    st.success(f"영상 저장 완료: {rel}")
            with c2:
                st.caption("저장 후 아래 'Skeleton 결과 영상 생성'에서 해당 영상을 선택할 수 있습니다.")
            with st.spinner("영상 미리보기 준비 중..."):
                preview_path = _orientation_safe_preview(
                    session_id,
                    slot,
                    uploaded,
                )
            st.video(str(preview_path))

    st.markdown("### 2) 저장된 원본 영상")
    videos = existing_videos(session_id)
    saved_video_meta = video_paths_for_session(session_id)
    if videos:
        for video in videos:
            st.write(f"- {video.name} ({video.stat().st_size / 1024 / 1024:.1f} MB)")
        if saved_video_meta:
            st.caption("Export CSV에는 아래 영상 경로가 함께 포함됩니다.")
            st.json(saved_video_meta, expanded=False)
    else:
        st.info("아직 저장된 영상이 없습니다.")
        return

    st.markdown("### 3) Skeleton 결과 영상 생성 / 다운로드")
    render_skeleton_output_notice("🟦 저장된 원본 영상에서 지표별 Skeleton Point와 참고 계산값이 표시된 결과 영상을 생성합니다. 생성된 MP4와 프레임별 CSV를 바로 다운로드할 수 있습니다.")
    status = dependencies_status()
    if not status["opencv"] or not status["mediapipe"]:
        st.warning("Skeleton 결과 영상 생성을 위해서는 OpenCV와 MediaPipe Pose backend가 필요합니다.")
        st.caption(dependency_message())
        return

    video_options = {video.name: video for video in videos}
    selected_video_name = st.selectbox("결과 영상을 생성할 원본 영상", list(video_options.keys()), key="result_video_source")
    metric_candidates = _metric_options_for_video(selected_video_name)
    metric_labels = {f"{m['display_name_kr']} · {m['metric_id']}": m for m in metric_candidates}
    selected_metric_label = st.selectbox("영상 위에 표시할 Skeleton/계산 항목", list(metric_labels.keys()), key="result_video_metric")
    metric = metric_labels[selected_metric_label]

    st.caption("선택한 지표에 필요한 Skeleton Point는 노란색으로 강조됩니다. 전체 포인트 표시를 켜면 모든 주요 포인트가 함께 표시됩니다.")
    default_start_sec, default_duration_sec = _default_analysis_window(session_id, selected_video_name)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1.25, 1, 1.25])
    with c1:
        start_sec = st.number_input("시작 시점(sec)", min_value=0.0, value=float(default_start_sec), step=0.5, key="result_video_start")
    with c2:
        duration_sec = st.number_input("처리 길이(sec)", min_value=0.5, max_value=60.0, value=float(default_duration_sec), step=0.5, key="result_video_duration")
    with c3:
        output_fps_label = st.selectbox(
            "결과 영상 출력 FPS",
            ["원본 FPS 자동", "15 fps", "12 fps", "10 fps", "6 fps"],
            key="result_video_output_fps_mode",
            help="분석 FPS는 업로드 영상의 실제 FPS를 자동 사용합니다. 이 옵션은 다운로드용 결과 영상 용량/부드러움만 조절합니다.",
        )
        output_fps = 0.0 if output_fps_label == "원본 FPS 자동" else float(output_fps_label.split()[0])
    with c4:
        show_all = st.checkbox("전체 포인트 표시", value=True, key="result_video_show_all")
    with c5:
        overlay_mode_label = st.selectbox("표시 방식", ["전체 요약 Overlay", "지표별 상세 Overlay"], key="result_overlay_mode")
    overlay_mode = "summary" if overlay_mode_label == "전체 요약 Overlay" else "detail"

    st.caption("v0.5.12부터 분석 FPS는 업로드 영상의 실제 메타데이터 FPS를 자동 사용하고, 거리 지표는 X/Y축 scale confidence를 분리 표시합니다. 위 FPS 옵션은 결과 영상 출력용이며, CSV raw/event 계산에는 원본 FPS가 적용됩니다.")
    if st.button("Skeleton 결과 영상 생성", type="primary", use_container_width=True):
        progress = st.progress(0.0, text="Skeleton 결과 영상 생성 중...")
        try:
            meta = create_overlay_video(
                session_id=session_id,
                video_path=video_options[selected_video_name],
                metric=metric,
                start_sec=start_sec,
                duration_sec=duration_sec,
                output_fps=output_fps,
                show_all=show_all,
                overlay_mode=overlay_mode,
                progress_callback=lambda p: progress.progress(p, text=f"Skeleton 결과 영상 생성 중... {int(p*100)}%"),
            )
            progress.progress(1.0, text="Skeleton 결과 영상 생성 완료")
            st.success("결과 영상 생성 완료")
            result_video_path = BASE_DIR / meta["result_video_path"]
            result_csv_path = BASE_DIR / meta["frame_stats_csv_path"]
            result_gif_path = BASE_DIR / meta.get("browser_preview_path", "") if meta.get("browser_preview_path") else None
            st.caption("앱 내 영상 플레이어입니다. 아래에서 같은 MP4 파일을 다운로드할 수도 있습니다.")
            _render_browser_mp4_player(result_video_path, key=f"player_{result_video_path.stem}")
            if result_gif_path and result_gif_path.exists():
                with st.expander("보조 GIF Preview 보기", expanded=False):
                    st.image(result_gif_path.read_bytes(), use_container_width=True)
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                _download_file_button(result_video_path, "결과 영상 MP4 다운로드", "video/mp4", key=f"download_video::{result_video_path.name}")
            with dc2:
                _download_file_button(result_csv_path, "프레임별 참고 데이터 CSV 다운로드", "text/csv", key=f"download_csv::{result_csv_path.name}")
            with dc3:
                if result_gif_path and result_gif_path.exists():
                    _download_file_button(result_gif_path, "브라우저 Preview GIF 다운로드", "image/gif", key=f"download_gif::{result_gif_path.name}")
            st.json(meta, expanded=False)
            _render_processed_feature_tables(meta, prefix="latest")
        except Exception as exc:
            progress.empty()
            st.error(f"Skeleton 결과 영상 생성 실패: {exc}")

    existing_results = processed_video_results(session_id)
    if existing_results:
        st.markdown("#### 기존 생성 결과 다운로드")
        for idx, meta in enumerate(reversed(existing_results[-5:])):
            result_video_path = BASE_DIR / meta.get("result_video_path", "")
            result_csv_path = BASE_DIR / meta.get("frame_stats_csv_path", "")
            result_gif_path = BASE_DIR / meta.get("browser_preview_path", "") if meta.get("browser_preview_path") else None
            with st.expander(f"{meta.get('created_at', '')} · {meta.get('metric_name_kr', meta.get('metric_id', ''))}", expanded=idx == 0):
                st.write(f"원본: {meta.get('source_video', '')}")
                st.write(f"처리 프레임: {meta.get('frames_processed', '')} / 분석 FPS: {meta.get('analysis_fps', meta.get('source_fps', ''))} / 결과 영상 FPS: {meta.get('output_fps', '')}")
                if result_video_path.exists():
                    _render_browser_mp4_player(result_video_path, key=f"old_player_{idx}_{result_video_path.stem}")
                if result_gif_path and result_gif_path.exists():
                    with st.expander("보조 GIF Preview 보기", expanded=False):
                        st.image(result_gif_path.read_bytes(), use_container_width=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    _download_file_button(result_video_path, "결과 영상 MP4 다운로드", "video/mp4", key=f"old_video::{idx}::{result_video_path.name}")
                with c2:
                    _download_file_button(result_csv_path, "프레임별 CSV 다운로드", "text/csv", key=f"old_csv::{idx}::{result_csv_path.name}")
                with c3:
                    if result_gif_path and result_gif_path.exists():
                        _download_file_button(result_gif_path, "Preview GIF 다운로드", "image/gif", key=f"old_gif::{idx}::{result_gif_path.name}")
                _render_processed_feature_tables(meta, prefix=f"old::{idx}")

    _render_frame_inspector(session_id, video_options, metric_candidates, selected_video_name)


def _metric_by_id(metric_id):
    for metric in metrics_defs["metrics"]:
        if metric["metric_id"] == metric_id:
            return metric
    return None


def _saved_video_options(session_id):
    videos = existing_videos(session_id)
    return {video.name: video for video in videos}




def _metric_options_for_video(video_name: str):
    """Return intuitive metric list based on side/rear video slot."""
    name = video_name.lower()
    if name.startswith("rear"):
        categories = ["rear_biomechanics"]
    elif name.startswith("side"):
        categories = [
            "side_biomechanics",
            "side_joint_rom",
            "side_manual_or_derived",
            "side_manual_label",
            "temporal_gait",
            "side_derived_later",
        ]
    else:
        categories = [
            "side_biomechanics",
            "side_joint_rom",
            "side_manual_or_derived",
            "side_manual_label",
            "temporal_gait",
            "rear_biomechanics",
        ]
    candidates = [m for m in metrics_defs["metrics"] if m.get("category") in categories]
    return candidates or metrics_defs["metrics"]



def _render_browser_mp4_player(path: Path, key: str, height: int = 520):
    """Render a browser video player using a data URI.

    Streamlit's st.video can show a black player for locally generated MP4s
    even when the file downloads correctly. A plain HTML5 video tag with an
    H.264/AVC source is more predictable for short generated preview videos.
    """
    if not path.exists():
        st.warning("생성된 영상 파일을 찾을 수 없습니다.")
        return
    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb > 45:
        st.info("영상 파일이 커서 앱 내 재생 대신 다운로드 확인을 권장합니다.")
        return
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        components.html(
            f"""
            <div style="width: 100%; background: #111; border-radius: 10px; padding: 8px; box-sizing: border-box;">
              <video id="{key}" controls playsinline preload="metadata" style="width: 100%; max-height: {height}px; background: #000; border-radius: 8px;">
                <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
                Your browser does not support the MP4 video tag.
              </video>
            </div>
            """,
            height=height + 34,
            scrolling=False,
        )
    except Exception as exc:
        st.warning(f"앱 내 영상 플레이어 표시 실패: {exc}")


def _download_file_button(path: Path, label: str, mime: str, key: str):
    if path.exists():
        st.download_button(
            label,
            path.read_bytes(),
            file_name=path.name,
            mime=mime,
            use_container_width=True,
            key=key,
        )


def _render_live_webrtc(metric):
    try:
        import av
        from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
    except Exception:
        st.info("Live Camera Stream은 선택 기능입니다. streamlit-webrtc/av 설치 환경에서 사용할 수 있습니다. 현재 환경에서는 업로드 영상 또는 카메라 스냅샷 Preview를 사용하세요.")
        return

    class PoseOverlayProcessor(VideoProcessorBase):
        def recv(self, frame):
            image = frame.to_ndarray(format="rgb24")
            try:
                overlay, _ = overlay_pose(Image.fromarray(image), metric=metric, show_all=False)
                return av.VideoFrame.from_ndarray(np.array(overlay), format="rgb24")
            except Exception:
                return frame

    try:
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(
            key=f"live-overlay-{metric['metric_id']}",
            video_processor_factory=PoseOverlayProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.caption("Live Stream은 촬영 상태 확인/라벨링 보조용입니다. 실시간 분석 성능 또는 최종 분석 정확도를 보장하지 않습니다.")
    except Exception as exc:
        st.warning(f"Live Stream 초기화가 되지 않았습니다: {exc}")


def tab_workflow_overlay():
    st.subheader("2. 촬영 순서 Wizard / Skeleton Overlay Preview")
    st.write("고객 문서의 순서에 맞춰 후면 정지 → 후면 달리기 → 측면 정지 → 측면 달리기 순서로 진행합니다. Live/Preview 값은 라벨링 보조용이며, 최종 학습 정답값은 MotionMetrix 직접 입력값입니다.")
    status = dependencies_status()
    if not status["python_preview_supported"]:
        st.error(
            f"현재 Python {status['python_version']} 환경입니다. Skeleton Preview는 MediaPipe 호환성을 위해 "
            "Python 3.9~3.12 가상환경에서 실행해 주세요. Windows에서는 setup_windows.bat 사용을 권장합니다."
        )
        st.caption("저장 영상 업로드, MotionMetrix 직접 입력, Export 기능은 계속 사용할 수 있습니다.")
    elif not status["mediapipe"] or not status["opencv"]:
        st.warning("Skeleton Overlay를 사용하려면 mediapipe와 opencv-python-headless가 필요합니다. Windows에서는 setup_windows.bat로 전용 가상환경을 생성하는 것을 권장합니다.")
        st.caption(dependency_message())

    session_id = st.session_state.get("current_session_id") or st.session_state.get(widget_key("session_id"))
    if not session_id:
        st.warning("먼저 새 세션을 생성하거나 세션 ID를 입력하세요.")
        return

    step_options = {step["title"]: step for step in WORKFLOW_STEPS}
    selected_step_title = st.selectbox("작업 순서 선택", list(step_options.keys()))
    step = step_options[selected_step_title]
    st.info(step["guide"])

    metric_candidates = [_metric_by_id(mid) for mid in step["metrics"]]
    metric_candidates = [m for m in metric_candidates if m]
    metric_labels = {f"{m['display_name_kr']} · {m['metric_id']}": m for m in metric_candidates}
    metric = metric_labels[st.selectbox("확인할 계산 항목", list(metric_labels.keys()))]

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown("#### Overlay 화면")
        preview_mode = st.radio("Preview 입력 방식", ["저장된 영상 프레임(권장)", "카메라 스냅샷(보조)", "Live Camera Stream(실험)"], horizontal=True)
        st.caption("기본 검수/사용은 저장된 측면·후면 영상을 업로드한 뒤 프레임 Preview를 생성하는 방식입니다. 카메라 스냅샷은 단일 자세 확인용, Live Stream은 브라우저 환경에 따라 달라지는 실험 기능입니다.")
        show_all_points = st.checkbox("전체 포인트 표시", value=False, help="꺼두면 선택 지표에 필요한 포인트만 강조합니다.")
        overlay_image = None
        stats = {}
        source_info = {"step_id": step["step_id"], "slot": step["slot"], "metric_id": metric["metric_id"], "metric_name_kr": metric["display_name_kr"]}

        if preview_mode == "저장된 영상 프레임(권장)":
            video_options = _saved_video_options(session_id)
            if not video_options:
                st.warning("먼저 영상 업로드 탭에서 측면/후면 영상을 저장하세요.")
            else:
                preferred = [name for name in video_options if name.startswith(step["slot"])]
                ordered_names = preferred + [name for name in video_options if name not in preferred]
                selected_video_name = st.selectbox("Preview 영상 선택", ordered_names)
                timestamp_sec = st.number_input("프레임 시점(sec)", min_value=0.0, value=0.0, step=0.1)
                if st.button("선택 프레임 Skeleton Preview 생성", type="primary"):
                    try:
                        frame = extract_frame(video_options[selected_video_name], timestamp_sec)
                        overlay_image, stats = overlay_pose(frame, metric=metric, show_all=show_all_points)
                        source_info.update({"source_type": "saved_video_frame", "source_video": str(video_options[selected_video_name].relative_to(BASE_DIR)), "timestamp_sec": timestamp_sec})
                    except Exception as exc:
                        st.error(f"Preview 생성 실패: {exc}")

        elif preview_mode == "카메라 스냅샷(보조)":
            captured = st.camera_input("카메라 스냅샷 촬영")
            if captured:
                try:
                    frame = Image.open(captured).convert("RGB")
                    overlay_image, stats = overlay_pose(frame, metric=metric, show_all=show_all_points)
                    source_info.update({"source_type": "camera_snapshot"})
                except Exception as exc:
                    st.error(f"스냅샷 처리 실패: {exc}")

        else:
            _render_live_webrtc(metric)
            st.caption("Live Stream은 브라우저/Streamlit Cloud 환경에 따라 동작이 달라질 수 있습니다. 검수 안정성은 저장된 영상 프레임 Preview 방식을 우선 권장합니다.")

        if overlay_image is not None:
            st.image(overlay_image, caption="Skeleton Overlay Preview", use_container_width=True)
            if st.button("현재 Preview 결과 저장", use_container_width=True):
                image_path = save_preview_result(session_id, overlay_image, {**source_info, "stats": stats})
                st.success(f"Preview 저장 완료: {image_path}")

    with c2:
        st.markdown("#### 계산정보 / 입력 가이드")
        render_metric_guide(metric, keypoints, derived)
        st.markdown("#### 현재 Preview 계산정보")
        if stats:
            st.json(stats, expanded=True)
        else:
            st.caption("Preview를 생성하면 현재 프레임 기준 참고 계산값이 표시됩니다.")
        st.markdown("#### MotionMetrix 입력 안내")
        st.info("고객 요청에 따라 촬영 Wizard에서는 MotionMetrix 입력 영역을 제거했습니다. MotionMetrix 값은 측면 입력/종합 입력 탭에서 평균값 중심으로 입력합니다. 후면 지표는 Skeleton-only 참고값으로 관리하며, 산출 가능한 후면 각도는 최종 비교표와 CSV에 함께 표시합니다.")

    st.markdown("#### 촬영 순서 체크리스트")
    checklist_rows = []
    saved_videos = video_paths_for_session(session_id)
    for item in WORKFLOW_STEPS:
        checklist_rows.append({
            "순서": item["title"],
            "영상 슬롯": item["slot"],
            "저장 여부": "완료" if saved_videos.get(f"{item['slot']}_video_path") else "미저장",
            "주요 확인 항목": ", ".join(item["metrics"]),
        })
    st.dataframe(pd.DataFrame(checklist_rows), hide_index=True, use_container_width=True)



def _field_map():
    return {field["field_id"]: field for field in all_session_fields(metrics_defs)}


def _render_selected_session_fields(field_ids: list[str], columns: int = 2):
    fields = _field_map()
    cols = st.columns(columns)
    visible = [fields[field_id] for field_id in field_ids if field_id in fields]
    for idx, field in enumerate(visible):
        with cols[idx % columns]:
            render_input(field, st.session_state.get(widget_key(field["field_id"])))


def tab_analysis_info():
    st.subheader("1. 분석 정보")
    st.write("환자 식별정보와 분석에 필요한 기본값을 입력합니다. 이름 대신 환자번호 또는 비식별 ID 사용을 권장합니다.")

    if not st.session_state.get("current_session_id"):
        if st.button("새 분석 세션 생성", type="primary", use_container_width=True):
            sid = _start_new_analysis_session()
            st.success(f"분석 ID가 생성되었습니다: {sid}")
        else:
            st.info("먼저 새 분석 세션을 생성하세요.")
            return

    _render_selected_session_fields(
        ["session_id", "user_id", "age", "sex", "height_cm", "weight_kg", "running_speed_kmh"],
        columns=2,
    )

    with st.expander("분석 구간 / 촬영 보정 설정 (선택)", expanded=False):
        st.caption("별도 측정 구간이 정해져 있지 않으면 업로드한 달리기 영상을 처음부터 최대 60초까지 분석합니다.")
        _render_selected_session_fields(
            [
                "measurement_date",
                "side_measure_start_sec",
                "side_measure_end_sec",
                "rear_measure_start_sec",
                "rear_measure_end_sec",
                "side_progress_direction",
                "side_ground_y_px",
                "rear_ground_y_px",
                "contact_threshold_px",
                "side_x_scale_mm_per_px",
                "side_y_scale_mm_per_px",
                "rear_x_scale_mm_per_px",
                "rear_y_scale_mm_per_px",
            ],
            columns=2,
        )

    with st.expander("MotionMetrix 기준값 입력 (선택 · 비교용)", expanded=False):
        st.caption("AI 분석에는 필수 입력이 아닙니다. 보유한 MotionMetrix 값을 입력하면 결과 화면과 Excel에서 비교값을 함께 확인할 수 있습니다.")
        render_metrics(
            metric_filter(
                [
                    "side_biomechanics",
                    "side_joint_rom",
                    "side_manual_label",
                    "temporal_gait",
                    "aggregate_motionmetrix",
                    "optional_motionmetrix",
                ]
            )
        )

    if st.button("분석 정보 저장", type="primary", use_container_width=True):
        sid, _ = save_current_session()
        st.success(f"분석 정보 저장 완료: {sid}")


def _saved_role_video(session_id: str, slot: str) -> Path | None:
    meta = video_paths_for_session(session_id)
    rel = meta.get(f"{slot}_video_path")
    if not rel:
        return None
    path = BASE_DIR / rel
    return path if path.exists() else None


def _render_uploaded_video_status(session_id: str):
    side = _saved_role_video(session_id, "side_running")
    rear = _saved_role_video(session_id, "rear_running")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 측면 달리기 영상 · 필수")
        if side:
            st.success(f"저장 완료 · {side.name}")
            with st.spinner("측면 영상 미리보기 준비 중..."):
                side_preview = _orientation_safe_preview(
                    session_id,
                    "side_running",
                    side,
                )
            st.video(str(side_preview))

        else:
            st.warning("측면 달리기 영상이 필요합니다.")
    with c2:
        st.markdown("#### 후면 달리기 영상 · 선택")
        if rear:
            st.success(f"저장 완료 · {rear.name}")
            with st.spinner("후면 영상 미리보기 준비 중..."):
                rear_preview = _orientation_safe_preview(
                    session_id,
                    "rear_running",
                    rear,
                )
            st.video(str(rear_preview))

        else:
            st.info("후면 지표 확인이 필요한 경우 업로드하세요.")


def tab_analysis_videos():
    st.subheader("2. 영상 업로드")
    st.write("측면 달리기 영상을 업로드하면 최종 AI 모델 분석을 실행할 수 있습니다. 후면 영상은 후면 자세 지표가 필요한 경우 추가합니다.")
    session_id = st.session_state.get("current_session_id")
    if not session_id:
        st.warning("먼저 '분석 정보'에서 새 분석을 시작하세요.")
        return

    upload_specs = [
        ("side_running", "측면 달리기 영상", True),
        ("rear_running", "후면 달리기 영상", False),
    ]

    for slot, label, required in upload_specs:
        suffix = "필수" if required else "선택"
        st.markdown(f"#### {label} · {suffix}")

        uploaded = st.file_uploader(
            f"{label} 업로드",
            type=["mp4"],
            key=f"final_upload::{slot}",
            help="MP4, 최대 300MB, 최대 60초 기준",
        )

        if uploaded is not None:
            with st.spinner("영상 미리보기 준비 중..."):
                preview_path = _orientation_safe_preview(
                    session_id,
                    slot,
                    uploaded,
                )
            st.video(str(preview_path))

            if st.button(
                f"{label} 저장",
                key=f"final_save::{slot}",
                use_container_width=True,
            ):
                filename = f"{slot}{Path(uploaded.name).suffix.lower()}"
                rel = save_uploaded_file(
                    session_id,
                    uploaded,
                    filename,
                    slot=slot,
                )
                save_current_session()
                st.success(f"저장 완료: {rel}")

    st.divider()
    _render_uploaded_video_status(session_id)
    st.caption("분석 실행 시 저장된 측면/후면 영상을 Pose/Skeleton feature로 변환한 뒤 Frozen v0.16 모델 결과를 생성합니다.")


def _final_analysis_window(session_id: str, slot: str) -> tuple[float, float]:
    try:
        meta = load_session(session_id).get("session_meta", {})
    except Exception:
        meta = {}
    prefix = "rear" if slot.startswith("rear") else "side"
    try:
        start = max(0.0, float(meta.get(f"{prefix}_measure_start_sec") or 0.0))
    except Exception:
        start = 0.0
    end_raw = meta.get(f"{prefix}_measure_end_sec")
    try:
        end = float(end_raw) if end_raw not in (None, "") else None
    except Exception:
        end = None
    if end is not None and end > start:
        return start, max(0.5, min(60.0, end - start))
    return start, 60.0


def _prepare_final_model_features(session_id: str, progress=None) -> list[dict]:
    side = _saved_role_video(session_id, "side_running")
    if side is None:
        raise RuntimeError("측면 달리기 영상이 없습니다. 먼저 영상을 업로드해 주세요.")

    jobs = [("side_running", side, "overstride")]
    rear = _saved_role_video(session_id, "rear_running")
    if rear is not None:
        jobs.append(("rear_running", rear, "pelvic_drop"))

    results = []
    total_jobs = len(jobs)
    for job_index, (slot, video_path, metric_id) in enumerate(jobs):
        metric = _metric_by_id(metric_id)
        if metric is None:
            raise RuntimeError(f"분석 지표 정의를 찾을 수 없습니다: {metric_id}")
        start_sec, duration_sec = _final_analysis_window(session_id, slot)

        def _progress(local_value):
            if progress is not None:
                overall = (job_index + float(local_value)) / max(total_jobs, 1)
                progress.progress(min(0.92, overall * 0.92), text=f"영상 분석 중 · {slot} · {int(float(local_value) * 100)}%")

        result = create_overlay_video(
            session_id=session_id,
            video_path=video_path,
            metric=metric,
            start_sec=start_sec,
            duration_sec=duration_sec,
            output_fps=10.0,
            show_all=True,
            progress_callback=_progress,
            overlay_mode="summary",
        )
        results.append(result)
    return results


def _run_final_model_analysis(session_id: str, progress=None) -> dict:
    from src.session_exports import export_session_debug_files

    save_current_session()
    _prepare_final_model_features(session_id, progress=progress)
    if progress is not None:
        progress.progress(0.94, text="Frozen v0.16 모델 분석 중...")

    debug_paths = export_session_debug_files(session_id)
    zip_paths = [path for path in debug_paths if path.suffix.lower() == ".zip"]
    if not zip_paths:
        raise RuntimeError("분석 입력 패키지를 생성하지 못했습니다.")

    mode, api_url, token = _inference_settings()
    if mode == "api" and not token:
        raise RuntimeError("API 모드에서는 RUNNINGAI_API_TOKEN 설정이 필요합니다.")

    result = analyze_session(zip_paths[-1], mode, api_url, token)
    saved_paths = save_analysis_result(session_id, result)
    csv_path = next(path for path in saved_paths if path.suffix.lower() == ".csv")
    parity = validate_result_export_parity(result, csv_path)
    if not parity["pass"]:
        raise RuntimeError(f"AI 결과/Export 일치 검증 실패: {parity['mismatches']}")

    # Final customer export is always regenerated after model inference so the
    # workbook contains the frozen-model result sheet, not only skeleton data.
    export_all(metrics_defs, visual_defs, session_id=session_id)
    sync_analysis_sheet(result)
    st.session_state[f"model_result::{session_id}"] = result

    if progress is not None:
        progress.progress(1.0, text="AI 분석 완료")
    return result


def _strike_display(value) -> str:
    mapping = {
        "heel": "리어풋 (Heel)",
        "midfoot": "미드풋 (Midfoot)",
        "forefoot": "포어풋 (Forefoot)",
    }
    if value is None or value == "":
        return "N/A"
    return mapping.get(str(value).lower(), str(value))


def _quality_display(value) -> str:
    mapping = {
        "completed": "측정 가능",
        "review_required": "검토 필요",
        "unmeasurable": "측정 불가",
    }
    return mapping.get(str(value or "").lower(), str(value or "N/A"))


def _format_result_value(value, unit="", digits=1):
    if value is None or value == "":
        return "N/A"
    try:
        text = f"{float(value):.{digits}f}"
    except Exception:
        text = str(value)
    return f"{text} {unit}".strip()


def _render_model_result(result: dict):
    overstride = result.get("overstride") or {}
    strike = result.get("strike_type") or {}
    quality = result.get("quality") or {}

    st.markdown("### 최종 AI 모델 결과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overstride", _format_result_value(overstride.get("prediction_mm"), "mm", 1))
    c2.metric("Strike Type", _strike_display(strike.get("final_class")))
    confidence = strike.get("final_confidence")
    c3.metric("Strike Confidence", "N/A" if confidence is None else f"{float(confidence) * 100:.1f}%")
    c4.metric("분석 품질", _quality_display(quality.get("status")))

    if quality.get("status") == "unmeasurable":
        reasons = quality.get("unmeasurable_reasons") or []
        st.error("현재 영상은 일부 또는 전체 지표를 안정적으로 측정하기 어렵습니다." + (f" 사유: {', '.join(map(str, reasons))}" if reasons else ""))
    elif strike.get("review_required") or overstride.get("review_required") or quality.get("status") == "review_required":
        st.warning("신뢰도 또는 영상 품질 기준에 따라 검토가 필요한 항목이 있습니다.")
    else:
        st.success("최종 Frozen 모델 분석이 완료되었습니다.")

    posture = result.get("posture_metrics") or {}
    metric_specs = [
        ("cadence_spm", "Cadence", "spm", 1),
        ("contact_time_ms", "Contact Time", "ms", 1),
        ("forward_lean_deg", "Forward Lean", "°", 1),
        ("max_thigh_flexion_deg", "Max Thigh Flexion", "°", 1),
        ("max_thigh_extension_deg", "Max Thigh Extension", "°", 1),
        ("hip_rom_deg", "Hip ROM", "°", 1),
        ("knee_flexion_touchdown_deg", "Knee Flexion @ Touchdown", "°", 1),
        ("max_knee_flexion_swing_deg", "Max Knee Flexion @ Swing", "°", 1),
        ("knee_rom_deg", "Knee ROM", "°", 1),
        ("vertical_displacement_mm", "Vertical Displacement", "mm", 1),
        ("pelvic_drop_deg", "Pelvic Drop", "°", 1),
        ("hip_hike_difference_deg", "Hip Hike Difference", "°", 1),
        ("shank_angle_touchdown_deg", "Shank Angle @ Touchdown", "°", 1),
    ]
    rows = []
    for key, label, unit, digits in metric_specs:
        if posture.get(key) is not None:
            rows.append({"Pose 기반 분석 지표": label, "결과": _format_result_value(posture.get(key), unit, digits)})
    if rows:
        st.markdown("### Pose 기반 자세 지표")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    model = result.get("model_info") or {}
    st.caption(f"Model: {model.get('model_version', 'final-20260825')} · 공식 Strike 결과는 bilateral patient-consensus final_class 기준")


def _render_model_motionmetrix_comparison(session_id: str, result: dict):
    try:
        values = load_session(session_id).get("motionmetrix_values", {})
    except Exception:
        values = {}

    rows = []
    overstride_mm = values.get("overstride_mean_mm")
    model_overstride = (result.get("overstride") or {}).get("prediction_mm")
    if overstride_mm not in (None, ""):
        try:
            reference = float(overstride_mm)
            prediction = float(model_overstride) if model_overstride is not None else None
            rows.append({
                "항목": "Overstride",
                "AI 모델 결과": None if prediction is None else round(prediction, 2),
                "MotionMetrix": round(reference, 2),
                "차이": None if prediction is None else round(prediction - reference, 2),
                "단위": "mm",
            })
        except Exception:
            pass

    strike_reference = values.get("session_representative_strike_type")
    if strike_reference not in (None, ""):
        rows.append({
            "항목": "Strike Type",
            "AI 모델 결과": _strike_display((result.get("strike_type") or {}).get("final_class")),
            "MotionMetrix": str(strike_reference),
            "차이": "일치" if str((result.get("strike_type") or {}).get("final_class", "")).lower() == str(strike_reference).lower() else "확인 필요",
            "단위": "",
        })

    if rows:
        st.markdown("### MotionMetrix 비교")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def tab_ai_analysis():
    st.subheader("3. AI 분석 결과")
    st.write("저장된 달리기 영상을 Pose/Skeleton feature로 변환한 뒤 Frozen v0.16 모델을 적용하여 최종 결과를 생성합니다.")
    session_id = st.session_state.get("current_session_id")
    if not session_id:
        st.warning("먼저 분석 정보를 입력하고 영상을 업로드하세요.")
        return

    side = _saved_role_video(session_id, "side_running")
    rear = _saved_role_video(session_id, "rear_running")
    c1, c2 = st.columns(2)
    c1.metric("측면 영상", "준비 완료" if side else "필요")
    c2.metric("후면 영상", "준비 완료" if rear else "선택 사항")

    if st.button("AI 분석 실행", type="primary", use_container_width=True, disabled=side is None):
        progress = st.progress(0.0, text="분석 준비 중...")
        try:
            result = _run_final_model_analysis(session_id, progress=progress)
            st.success("영상 분석과 최종 모델 추론이 완료되었습니다.")
        except Exception as exc:
            progress.empty()
            st.error(f"AI 분석 실패: {exc}")

    result = st.session_state.get(f"model_result::{session_id}") or load_saved_analysis(session_id)
    if not result:
        st.info("'AI 분석 실행'을 누르면 최종 모델 결과가 이 화면에 표시됩니다.")
        return

    _render_model_result(result)
    _render_model_motionmetrix_comparison(session_id, result)

    with st.expander("분석 과정 / 품질 정보", expanded=False):
        st.json({
            "quality": result.get("quality") or {},
            "model_info": result.get("model_info") or {},
            "source": result.get("source") or {},
        }, expanded=True)


def _primary_result_downloads(session_id: str) -> list[Path]:
    return [
        EXPORTS_DIR / "final_comparison_summary.xlsx",
        EXPORTS_DIR / "final_comparison_summary.csv",
        EXPORTS_DIR / f"{session_id}_api_analysis_result.csv",
        EXPORTS_DIR / f"{session_id}_api_analysis_result.json",
    ]


def tab_final_results_export():
    st.subheader("4. 결과 확인 / Export")
    st.write("최종 모델 결과와 비교 결과를 확인하고 납품·검수용 파일을 다운로드합니다.")
    session_id = st.session_state.get("current_session_id")
    if not session_id:
        st.warning("분석 세션이 없습니다.")
        return

    result = st.session_state.get(f"model_result::{session_id}") or load_saved_analysis(session_id)
    if not result:
        st.warning("먼저 'AI 분석 결과' 탭에서 최종 모델 분석을 실행하세요.")
        return

    _render_model_result(result)
    _render_model_motionmetrix_comparison(session_id, result)

    st.markdown("### 주요 결과 다운로드")
    workbook_path = EXPORTS_DIR / "final_comparison_summary.xlsx"
    if not workbook_path.exists():
        export_all(metrics_defs, visual_defs, session_id=session_id)
    sync_analysis_sheet(result)
    for path in _primary_result_downloads(session_id):
        if not path.exists():
            continue
        if path.suffix == ".xlsx":
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif path.suffix == ".json":
            mime = "application/json"
        else:
            mime = "text/csv"
        label_map = {
            "final_comparison_summary.xlsx": "최종 분석 결과 Excel",
            "final_comparison_summary.csv": "Skeleton / MotionMetrix 비교 CSV",
            f"{session_id}_api_analysis_result.csv": "AI 모델 결과 CSV",
            f"{session_id}_api_analysis_result.json": "AI 모델 결과 JSON",
        }
        st.download_button(
            label_map.get(path.name, path.name),
            path.read_bytes(),
            file_name=path.name,
            mime=mime,
            use_container_width=True,
            key=f"final_download::{path.name}",
        )

    with st.expander("Skeleton / Debug 상세 정보", expanded=False):
        st.caption("최종 결과의 추적·검증을 위한 보조 정보입니다. 일반 분석 사용에서는 확인하지 않아도 됩니다.")
        _render_export_readiness(session_id)
        _render_final_comparison_table(session_id)

        extra_downloads = [
            EXPORTS_DIR / f"{session_id}_session_all_skeleton_frames.csv",
            EXPORTS_DIR / f"{session_id}_session_gait_events.csv",
            EXPORTS_DIR / f"{session_id}_session_skeleton_metric_summary.csv",
        ]
        extra_downloads.extend(sorted(EXPORTS_DIR.glob(f"{session_id}_debug_export_package_*.zip"), reverse=True)[:1])
        for path in extra_downloads:
            if path.exists():
                mime = "application/zip" if path.suffix == ".zip" else "text/csv"
                st.download_button(
                    path.name,
                    path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    use_container_width=True,
                    key=f"debug_download::{path.name}",
                )

def tab_visual_labels():
    st.subheader("6. 육안 자세 평가")
    st.write("강도, 방향, 메모를 입력합니다. 이 값은 MotionMetrix 정답값과 분리 저장됩니다.")
    strength_options = [x["value"] for x in visual_defs["strength_options"]]
    direction_options = [x["value"] for x in visual_defs["direction_options"]]
    for label in visual_defs["labels"]:
        with st.expander(label["display_name_kr"], expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                fid = f"{label['label_id']}_strength"
                current = st.session_state.get(widget_key(fid), "none")
                st.selectbox("강도", strength_options, index=strength_options.index(current) if current in strength_options else 0, key=widget_key(fid), format_func=option_label)
            with c2:
                fid = f"{label['label_id']}_direction"
                current = st.session_state.get(widget_key(fid), "not_applicable")
                st.selectbox("방향", direction_options, index=direction_options.index(current) if current in direction_options else 0, key=widget_key(fid), format_func=option_label)
            fid = f"{label['label_id']}_memo"
            st.text_area("메모", value=st.session_state.get(widget_key(fid), ""), key=widget_key(fid), height=80)
    if st.button("육안 평가 저장", type="primary"):
        sid, _ = save_current_session()
        st.success(f"저장 완료: {sid}")


def tab_skeleton_guide():
    st.subheader("7. Skeleton Point Guide")
    st.write("고객이 입력하는 각 지표가 어떤 포인트와 계산 기준으로 연결되는지 확인하는 화면입니다.")
    point_df = pd.DataFrame(keypoint_defs["keypoints"])
    st.markdown("#### 전체 포인트")
    st.dataframe(point_df[["id", "name_kr", "name_en", "group"]], hide_index=True, use_container_width=True)
    metric_options = {f"{m['display_name_kr']} · {m['metric_id']}": m for m in metrics_defs["metrics"]}
    selected = st.selectbox("지표 선택", list(metric_options.keys()))
    metric = metric_options[selected]
    c1, c2 = st.columns([1, 1])
    with c1:
        render_skeleton_graph(metric.get("required_points", []))
    with c2:
        render_metric_guide(metric, keypoints, derived)



def _render_export_readiness(session_id: str):
    from src.session_exports import latest_csv_by_role
    frame_sources = latest_csv_by_role(session_id, "frame_metrics")
    event_sources = latest_csv_by_role(session_id, "gait_events")
    clip_sources = latest_csv_by_role(session_id, "clip_summary")
    rows = []
    for role, label in [("rear_running", "후방 running Skeleton"), ("side_running", "측방 running Skeleton")]:
        rows.append({
            "체크 항목": label,
            "상태": "생성 완료" if role in frame_sources else "미생성",
            "프레임 CSV": frame_sources.get(role, {}).get("relative_path", ""),
            "이벤트 CSV": event_sources.get(role, {}).get("relative_path", ""),
            "요약 CSV": clip_sources.get(role, {}).get("relative_path", ""),
        })
    st.markdown("#### Export 전 Skeleton source 확인")
    st.caption("최종 비교표의 running 지표는 side_running, 후면 지표는 rear_running 결과만 참조합니다. static 결과는 baseline/preview 용도이며 running 지표 source로 사용하지 않습니다.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)



def _inference_settings():
    try:
        secrets = st.secrets
    except Exception:
        secrets = None
    return resolve_inference_settings(secrets)


def _render_model_analysis(session_id: str):
    from src.session_exports import export_session_debug_files

    st.markdown("#### v0.16 모델 분석")
    mode, api_url, token = _inference_settings()
    mode_label = "Embedded" if mode == "embedded" else "API"

    c1, c2 = st.columns(2)
    with c1:
        if st.button("모델 상태 확인", use_container_width=True, key=f"model_status::{session_id}"):
            try:
                info = inference_status(mode, api_url, token)
                st.success(f"{info.get('model_version', '')} · {mode_label}")
            except Exception as exc:
                st.warning(f"모델 상태 확인 실패: {exc}")
    with c2:
        run_disabled = mode == "api" and not bool(token)
        if st.button(
            "현재 세션 분석",
            use_container_width=True,
            disabled=run_disabled,
            key=f"model_analyze::{session_id}",
        ):
            try:
                sid, _ = save_current_session()
                paths = export_session_debug_files(sid)
                zip_paths = [path for path in paths if path.suffix.lower() == ".zip"]
                if not zip_paths:
                    raise RuntimeError("Debug Export ZIP이 생성되지 않았습니다.")
                result = analyze_session(zip_paths[-1], mode, api_url, token)
                saved_paths = save_analysis_result(sid, result)
                parity = validate_result_export_parity(
                    result,
                    next(path for path in saved_paths if path.suffix.lower() == ".csv"),
                )
                sync_analysis_sheet(result)
                st.session_state[f"model_result::{sid}"] = result
                if parity["pass"]:
                    st.success("분석 및 Export 저장 완료")
                else:
                    st.error(f"결과/Export 일치 검증 실패: {parity['mismatches']}")
            except Exception as exc:
                st.error(f"분석 실패: {exc}")

    if mode == "api" and not token:
        st.caption("API 모드에서는 RUNNINGAI_API_TOKEN 설정이 필요합니다.")

    result = st.session_state.get(f"model_result::{session_id}") or load_saved_analysis(session_id)
    if not result:
        return

    overstride = result.get("overstride") or {}
    strike = result.get("strike_type") or {}
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Overstride",
        "N/A" if overstride.get("prediction_mm") is None else f"{float(overstride['prediction_mm']):.1f} mm",
    )
    m2.metric("Strike Type", strike.get("final_class") or "N/A")
    confidence = strike.get("final_confidence")
    m3.metric("Strike Confidence", "N/A" if confidence is None else f"{float(confidence):.3f}")
    st.dataframe(pd.DataFrame(analysis_summary_rows(result)), hide_index=True, use_container_width=True)
    if strike.get("review_required") or overstride.get("review_required"):
        st.warning("검토 필요 항목이 포함되어 있습니다.")
    model = result.get("model_info") or {}
    st.caption(f"Model: {model.get('model_version', 'final-20260825')} · Mode: {mode_label}")


def tab_review_export():
    st.subheader("8. 최종 검토 / Export")
    meta, values, visual = collect_session()
    session_id = st.session_state.get("current_session_id") or meta.get("session_id")
    done, total = completion_summary(meta, values, visual)
    progress = done / total if total else 0
    st.progress(progress, text=f"필수 입력 완료율: {done}/{total}")
    missing = missing_values_preview(session_id or "CURRENT", meta, values, visual)
    if missing:
        st.warning(f"필수 누락 {len(missing)}개가 있습니다.")
        st.dataframe(pd.DataFrame(missing), hide_index=True, use_container_width=True)
    else:
        st.success("현재 입력값 기준으로 필수 항목이 모두 입력되었습니다.")

    if session_id:
        _render_export_readiness(session_id)
        _render_final_comparison_table(session_id)
        _render_model_analysis(session_id)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("현재 세션 저장", type="primary", use_container_width=True):
            sid, review = save_current_session()
            st.success(f"저장 완료: {sid}")
    with c2:
        if st.button("전체 Export 생성", use_container_width=True):
            sid, _ = save_current_session()
            paths = export_all(metrics_defs, visual_defs, session_id=sid)
            st.success(f"현재 세션 저장 완료: {sid}")
            st.success("현재 세션 기준 Export 생성 완료. 아래 CSV/Excel을 다운로드하세요.")
            for path in paths:
                st.write(f"- {path.relative_to(BASE_DIR)}")
            quality_path = EXPORTS_DIR / "data_quality_summary.csv"
            if quality_path.exists():
                st.markdown("#### 데이터 품질 요약")
                st.dataframe(pd.read_csv(quality_path), hide_index=True, use_container_width=True)
    with c3:
        if st.button("전체 백업 ZIP 생성", use_container_width=True):
            zip_path = make_backup_zip()
            st.success(f"백업 생성: {zip_path.name}")
    st.markdown("#### 다운로드")
    export_downloads = [
        EXPORTS_DIR / "final_comparison_summary.xlsx",
        EXPORTS_DIR / "final_comparison_summary.csv",
        EXPORTS_DIR / "training_dataset_wide.csv",
        EXPORTS_DIR / "training_dataset_long.csv",
        EXPORTS_DIR / "metric_dictionary.csv",
        EXPORTS_DIR / "missing_value_report.csv",
        EXPORTS_DIR / "data_quality_report.csv",
        EXPORTS_DIR / "data_quality_summary.csv",
    ]
    if session_id:
        export_downloads.extend([
            EXPORTS_DIR / f"{session_id}_session_all_skeleton_frames.csv",
            EXPORTS_DIR / f"{session_id}_session_gait_events.csv",
            EXPORTS_DIR / f"{session_id}_session_skeleton_metric_summary.csv",
        ])
        export_downloads.extend(sorted(EXPORTS_DIR.glob(f"{session_id}_debug_export_package_*.zip"), reverse=True)[:1])
        export_downloads.extend([
            EXPORTS_DIR / f"{session_id}_api_analysis_result.json",
            EXPORTS_DIR / f"{session_id}_api_analysis_result.csv",
        ])
    for path in export_downloads:
        if path.exists():
            if path.suffix == ".xlsx":
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif path.suffix == ".json":
                mime = "application/json"
            elif path.suffix == ".zip":
                mime = "application/zip"
            else:
                mime = "text/csv"
            st.download_button(path.name, path.read_bytes(), file_name=path.name, mime=mime, use_container_width=True)
    backups = sorted(EXPORTS_DIR.glob("labeling_data_backup_*.zip"), reverse=True)
    if backups:
        latest = backups[0]
        st.download_button("최근 백업 ZIP 다운로드", latest.read_bytes(), file_name=latest.name, mime="application/zip", use_container_width=True)


def main():
    init_state()
    render_sidebar()
    st.title("달리기 자세 AI 분석")
    st.caption("달리기 영상 → Pose/Skeleton 분석 → Frozen v0.16 모델 → 최종 결과")

    st.info("측면 달리기 영상을 업로드한 뒤 'AI 분석 실행'을 누르면 Overstride와 Strike Type을 포함한 최종 모델 결과가 생성됩니다. MotionMetrix 입력은 비교가 필요한 경우에만 선택적으로 사용합니다.")

    tabs = st.tabs([
        "1. 분석 정보",
        "2. 영상 업로드",
        "3. AI 분석 결과",
        "4. 결과 / Export",
    ])
    with tabs[0]:
        tab_analysis_info()
    with tabs[1]:
        tab_analysis_videos()
    with tabs[2]:
        tab_ai_analysis()
    with tabs[3]:
        tab_final_results_export()


if __name__ == "__main__":
    main()
