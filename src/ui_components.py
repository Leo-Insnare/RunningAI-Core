import pandas as pd
import streamlit as st


def widget_key(field_id):
    return f"field::{field_id}"


OPTION_LABELS = {
    "male": "남성",
    "female": "여성",
    "other": "기타",
    "unknown": "미확인",
    "normal": "정상",
    "mild_asymmetry": "약한 비대칭",
    "clear_asymmetry": "뚜렷한 비대칭",
    "severe_asymmetry": "심한 비대칭",
    "heel": "힐 착지",
    "midfoot": "미드풋 착지",
    "forefoot": "포어풋 착지",
    "mixed": "혼합",
    "none": "없음",
    "mild": "약함",
    "clear": "뚜렷함",
    "severe": "심함",
    "left": "왼쪽",
    "right": "오른쪽",
    "both": "양쪽",
    "not_applicable": "해당 없음",
    "FV": "FV",
    "bodyweight_fraction": "체중 대비 비율",
    "N": "N",
    "motionmetrix_raw": "MotionMetrix 표시 단위 그대로",
    "steps/min": "steps/min",
    "strides/min": "strides/min",
    "cm": "cm",
    "mm": "mm",
    "ms": "ms",
    "sec": "sec",
    "m": "m",
    "left_to_right": "왼쪽 → 오른쪽",
    "right_to_left": "오른쪽 → 왼쪽",
}


FIELD_VALIDATION = {
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
    "side_ground_y_px": (0, 5000),
    "rear_ground_y_px": (0, 5000),
    "contact_threshold_px": (1, 200),
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


def option_label(value):
    return OPTION_LABELS.get(value, str(value))


def validation_range(field_id):
    if field_id in FIELD_VALIDATION:
        return FIELD_VALIDATION[field_id]
    for keyword, value_range in KEYWORD_VALIDATION:
        if keyword in field_id:
            return value_range
    return None


def _show_validation(field, value):
    value_range = validation_range(field["field_id"])
    if value_range is None or value in [None, ""]:
        return
    low, high = value_range
    st.caption(f"권장 확인 범위: {low} ~ {high}")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return
    if numeric_value < low or numeric_value > high:
        st.warning("입력값이 일반적인 확인 범위를 벗어났습니다. MotionMetrix 원본값/단위를 다시 확인하세요.", icon="⚠️")




def render_direct_input_style():
    """Inject compact styles that make customer-entered fields visually obvious."""
    st.markdown(
        """
        <style>
        .direct-input-notice {
            border-left: 6px solid #f97316;
            background: #fff7ed;
            padding: 0.7rem 0.9rem;
            border-radius: 0.55rem;
            margin: 0.35rem 0 0.75rem 0;
            font-weight: 600;
            color: #7c2d12;
        }
        .optional-input-notice {
            border-left: 6px solid #9ca3af;
            background: #f9fafb;
            padding: 0.6rem 0.8rem;
            border-radius: 0.55rem;
            margin: 0.25rem 0 0.6rem 0;
            color: #374151;
        }
        .skeleton-output-panel {
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            padding: 0.85rem 1rem;
            border-radius: 0.7rem;
            margin: 0.5rem 0 0.9rem 0;
            color: #1e3a8a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_direct_input_notice(text="🟧 주황색 표시는 고객이 MotionMetrix 값을 직접 입력해야 하는 핵심 영역입니다."):
    st.markdown(f'<div class="direct-input-notice">{text}</div>', unsafe_allow_html=True)


def render_skeleton_output_notice(text="🟦 저장된 영상에서 Skeleton Overlay 결과 영상을 생성하고 다운로드할 수 있습니다."):
    st.markdown(f'<div class="skeleton-output-panel">{text}</div>', unsafe_allow_html=True)


def render_input(field, value=None, key_suffix=None):
    key = widget_key(field["field_id"])
    if key_suffix:
        key = f"{key}::{key_suffix}"
    base_label = field["label_kr"]
    unit = field.get("unit", "")
    if unit:
        base_label = f"{base_label} ({unit})"
    typ = field.get("type", "text")
    is_required = bool(field.get("required"))
    required = "필수" if is_required else "선택"
    prefix = "🟧 직접 입력" if is_required else "⬜ 선택 입력"
    label = f"{prefix} · {base_label}"
    help_text = f"{required} 입력 항목입니다. 주황색 표시는 고객이 직접 입력해야 하는 핵심 데이터입니다." if is_required else f"{required} 입력 항목"
    extra_help = field.get("help_kr") or field.get("ui_help_kr") or ""
    if extra_help:
        help_text = f"{help_text}\n\n{extra_help}"

    if typ == "number":
        try:
            current = float(value) if value not in [None, ""] else None
        except (TypeError, ValueError):
            current = None
        result = st.number_input(label, value=current, step=0.1, key=key, placeholder="값 입력", help=help_text)
        _show_validation(field, result)
        return result

    if typ == "select":
        options = field.get("options", []) or [""]
        current = value if value in options else options[0]
        return st.selectbox(
            label,
            options,
            index=options.index(current),
            key=key,
            format_func=option_label,
            help=help_text,
        )

    if typ == "date":
        return st.text_input(label, value="" if value is None else str(value), key=key, placeholder="YYYY-MM-DD", help=help_text)

    return st.text_input(label, value="" if value is None else str(value), key=key, help=help_text)


def render_metric_guide(metric, keypoint_map=None, derived_map=None):
    points = metric.get("required_points", [])
    derived = metric.get("derived_points", [])
    with st.expander("사용 Skeleton Point / 계산 기준", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**사용 포인트**")
            if points and keypoint_map:
                point_rows = []
                for p in points:
                    item = keypoint_map.get(str(p), {})
                    point_rows.append({"Point": p, "의미": item.get("name_kr", "")})
                st.dataframe(pd.DataFrame(point_rows), use_container_width=True, hide_index=True)
            else:
                st.write("직접 사용 포인트 없음. 여러 지표를 종합해 사용하는 항목입니다.")
        with c2:
            st.markdown("**파생 포인트 / 계산 요소**")
            if derived and derived_map:
                derived_rows = []
                for d in derived:
                    item = derived_map.get(str(d), {})
                    derived_rows.append({"ID": d, "정의": item.get("definition_kr", d)})
                st.dataframe(pd.DataFrame(derived_rows), use_container_width=True, hide_index=True)
            else:
                st.write("별도 파생 포인트 없음")
        st.markdown("**계산 기준**")
        st.write(metric.get("calculation_basis_kr", ""))
        st.markdown("**입력 도움말**")
        st.info(metric.get("ui_help_kr", ""))
        st.caption("Skeleton 계산값은 라벨링/비교 보조값이며, MotionMetrix 입력값은 3차 모델링 정답값으로 분리 저장합니다. 후면 지표는 고객 피드백 기준 Skeleton-only로 관리합니다.")


def render_skeleton_graph(active_points=None):
    active_points = set(str(p) for p in (active_points or []))
    edges = [
        ("7", "11"), ("8", "12"), ("11", "12"),
        ("11", "13"), ("13", "15"), ("12", "14"), ("14", "16"),
        ("11", "23"), ("12", "24"), ("23", "24"),
        ("23", "25"), ("25", "27"), ("27", "29"), ("27", "31"),
        ("24", "26"), ("26", "28"), ("28", "30"), ("28", "32"),
    ]
    nodes = sorted({n for e in edges for n in e}, key=lambda x: int(x))
    lines = ["graph G {", "rankdir=TB;", "node [shape=circle, style=filled, fontname=\"Arial\"];", "edge [color=gray50];"]
    for n in nodes:
        color = "#ffd166" if n in active_points else "#edf2f4"
        lines.append(f'"{n}" [label="{n}", fillcolor="{color}"];')
    for a, b in edges:
        lines.append(f'"{a}" -- "{b}";')
    lines.append("}")
    st.graphviz_chart("\n".join(lines), use_container_width=True)


def status_badge(value):
    if value:
        return "완료"
    return "미입력"
