from __future__ import annotations

import json
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.config import MODEL_VERSION, REVIEW_CONFIDENCE_THRESHOLD
from app.model_engine import FrozenModelEngine


def main():
    engine = FrozenModelEngine(device="cpu")
    status = engine.status
    parity = json.loads((Path(__file__).resolve().parents[1] / "FINAL_MODEL_PARITY.json").read_text(encoding="utf-8"))
    assert status.model_version == "final-20260825"
    assert MODEL_VERSION == "final-20260825"
    assert abs(REVIEW_CONFIDENCE_THRESHOLD - 0.41) < 1e-12
    assert parity["pass"] is True
    print("Final API model verification: PASS")
    print("Model version:", MODEL_VERSION)
    print("Strike review threshold:", REVIEW_CONFIDENCE_THRESHOLD)
    print("Artifact integrity:", status.artifact_integrity)


if __name__ == "__main__":
    main()
