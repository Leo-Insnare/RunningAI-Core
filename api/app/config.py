from __future__ import annotations

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("RUNNINGAI_MODEL_DIR", ROOT_DIR / "models"))
STATUS_PATH = Path(os.environ.get("RUNNINGAI_STATUS_PATH", ROOT_DIR / "config" / "model_status.json"))
TOKEN_FILE = Path(os.environ.get("RUNNINGAI_TOKEN_FILE", ROOT_DIR / ".api_token"))

MODEL_VERSION = "final-20260825"
API_VERSION = "v1.4.0"
REVIEW_CONFIDENCE_THRESHOLD = 0.41
CRITICAL_SEQUENCE_CHANNELS = [
    "knee_flexion_deg",
    "shank_angle_abs_deg",
    "knee_flexion_delta_ic_deg",
    "heel_toe_y_diff_norm",
    "heel_ground_gap_norm",
    "ankle_x_rel_norm",
    "shank_angle_delta_ic_deg",
    "heel_x_rel_norm",
]
EXPECTED_ARTIFACT_SHA256 = {
    "strike_anchor_supcon_15model_v0_16.pt": "ed5ed6e8b28c5cdc76cfb7ce8a400d8c5449eb23e5c0a472c2a119070abfa66f",
    "strike_forefoot_heel_rescue_5model_v0_16.joblib": "34847790c04e5e551d6091a938d35def00a286e204dad18f695a2bd16b5f84ed",
    "overstride_cv_ensemble_candidate_v0_16.joblib": "a991fdb4e452154a645b2964e570273bf560a52466e7e8f76b3bbdcca70a8f6e",
    "overstride_final_inference_v1.json": "28620dd49f639c829656a2dd95e32a373805600a88ae556e1064e0a955ddfcf3",
}


def load_status() -> dict:
    if not STATUS_PATH.exists():
        return {
            "model_version": MODEL_VERSION,
            "final_independent_test_completed": False,
            "final_independent_targets_met": None,
            "final_independent_fingerprint": None,
        }
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
