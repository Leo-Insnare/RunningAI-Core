import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFINITION_DIR = BASE_DIR / "data" / "metric_definitions"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_definitions():
    for name in ["metric_definitions_v0_5_11.json", "metric_definitions_v0_5_10.json", "metric_definitions_v0_5_9.json", "metric_definitions_v0_5_8.json", "metric_definitions_v0_5_7.json", "metric_definitions_v0_5_6.json", "metric_definitions_v0_5_5.json", "metric_definitions_v0_5_4.json", "metric_definitions_v0_5_3.json", "metric_definitions_v0_4_8.json", "metric_definitions_v0_4_5.json", "metric_definitions_v0_4.json", "metric_definitions_v0_3.json", "metric_definitions_v0_2.json", "metric_definitions_v0_1.json"]:
        metric_path = DEFINITION_DIR / name
        if metric_path.exists():
            break
    metrics = load_json(metric_path)
    keypoints = load_json(DEFINITION_DIR / "keypoint_registry.json")
    visual = load_json(DEFINITION_DIR / "visual_label_definitions.json")
    return metrics, keypoints, visual


def metric_categories(metrics):
    groups = {}
    for metric in metrics["metrics"]:
        groups.setdefault(metric["category"], []).append(metric)
    return groups


def all_metric_fields(metrics):
    fields = []
    for metric in metrics["metrics"]:
        for field in metric.get("fields", []):
            fields.append({**field, "metric_id": metric["metric_id"], "metric": metric})
    return fields


def all_session_fields(metrics):
    return metrics.get("session_meta", [])


def all_visual_fields(visual_defs):
    fields = []
    for label in visual_defs["labels"]:
        base = label["label_id"]
        fields.extend([
            {"field_id": f"{base}_strength", "label_kr": f"{label['display_name_kr']} - 강도", "type": "select", "required": True},
            {"field_id": f"{base}_direction", "label_kr": f"{label['display_name_kr']} - 방향", "type": "select", "required": True},
            {"field_id": f"{base}_memo", "label_kr": f"{label['display_name_kr']} - 메모", "type": "text", "required": False},
        ])
    return fields


def keypoint_lookup(keypoints):
    return {str(item["id"]): item for item in keypoints["keypoints"]}


def derived_lookup(keypoints):
    return {item["id"]: item for item in keypoints["derived_points"]}
