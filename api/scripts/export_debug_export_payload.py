from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.debug_export_adapter import DebugExportAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("debug_export")
    parser.add_argument("--output", default="analyze_request.json")
    args = parser.parse_args()

    result = DebugExportAdapter.from_path(args.debug_export).build()
    Path(args.output).write_text(
        json.dumps(result.request.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
