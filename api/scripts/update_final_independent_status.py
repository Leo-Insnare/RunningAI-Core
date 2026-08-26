from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "config" / "model_status.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_json", type=Path)
    args = parser.parse_args()

    gate = json.loads(args.gate_json.read_text(encoding="utf-8"))
    if gate.get("version") != "final":
        raise SystemExit("Unexpected Final Independent gate version.")
    if not gate.get("frozen_artifact_integrity"):
        raise SystemExit("Frozen artifact integrity did not pass.")

    out = {
        "model_version": "final-20260825",
        "final_independent_test_completed": True,
        "final_independent_targets_met": bool(gate.get("modeling_core_targets_met")),
        "final_independent_fingerprint": gate.get("independent_fingerprint"),
        "strike_target_met": bool(gate.get("strike_target_met")),
        "overstride_target_met": bool(gate.get("overstride_target_met")),
    }
    STATUS.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Updated:", STATUS)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
