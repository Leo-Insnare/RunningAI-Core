from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app.video_analysis import MAX_VIDEO_DURATION_SEC, build_motionmetrix_comparison
from api.tests.fixture_debug_export import build_debug_export_bytes
from src.api_integration import (
    analysis_summary_rows,
    analyze_debug_export,
    analyze_debug_export_embedded,
    model_info,
    save_analysis_result,
    sync_analysis_sheet,
    validate_result_export_parity,
)


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return int(port)


def wait_url(url: str, timeout_sec: int = 60) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def same_result(a: dict, b: dict) -> bool:
    return (
        a.get("overstride", {}).get("prediction_mm")
        == b.get("overstride", {}).get("prediction_mm")
        and a.get("strike_type", {}).get("final_class")
        == b.get("strike_type", {}).get("final_class")
        and a.get("strike_type", {}).get("feet")
        == b.get("strike_type", {}).get("feet")
        and a.get("posture_metrics") == b.get("posture_metrics")
        and a.get("quality") == b.get("quality")
    )


def main() -> int:
    checks: dict[str, bool] = {}
    token = "integration-test-token"

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        zip_path = temp_dir / "fixture.zip"
        zip_path.write_bytes(build_debug_export_bytes())

        embedded = analyze_debug_export_embedded(zip_path)
        saved = save_analysis_result("fixture", embedded, export_dir=temp_dir)
        csv_path = next(path for path in saved if path.suffix == ".csv")
        checks["embedded_export_parity"] = bool(
            validate_result_export_parity(embedded, csv_path).get("pass")
        )

        workbook = temp_dir / "final_comparison_summary.xlsx"
        pd.DataFrame([{"metric": "fixture"}]).to_excel(
            workbook, index=False, sheet_name="Skeleton_vs_MotionMetrix"
        )
        sync_analysis_sheet(embedded, workbook)
        with pd.ExcelFile(workbook) as xls:
            checks["workbook_model_sheet"] = "AI_Model_Result" in xls.sheet_names
            model_sheet = pd.read_excel(xls, sheet_name="AI_Model_Result")
        checks["workbook_model_value"] = (
            str(model_sheet.iloc[0]["strike_final_class"])
            == str(embedded["strike_type"]["final_class"])
        )

        rows = analysis_summary_rows(embedded)
        overstride_row = next(row for row in rows if row["구분"] == "Overstride")
        strike_row = next(
            row
            for row in rows
            if row["구분"] == "Strike Type" and row["측면"] == "최종"
        )
        checks["ui_mapping_overstride"] = (
            overstride_row["결과"] == embedded["overstride"]["prediction_mm"]
        )
        checks["ui_mapping_strike"] = (
            strike_row["결과"] == embedded["strike_type"]["final_class"]
        )

        posture = embedded.get("posture_metrics") or {}
        checks["posture_metric_contract"] = all(
            key in posture
            for key in [
                "hip_rom_deg",
                "max_knee_flexion_swing_deg",
                "knee_rom_deg",
                "vertical_displacement_mm",
            ]
        )
        checks["duration_limit_60"] = (
            MAX_VIDEO_DURATION_SEC == 60.0
            and "max_value=60.0" in (ROOT / "app.py").read_text(encoding="utf-8")
            and "min(float(duration_sec or 5.0), 60.0)"
            in (ROOT / "src" / "pose_preview.py").read_text(encoding="utf-8")
        )

        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        main_source = app_source[app_source.rfind("def main():") :]
        checks["final_output_ui_contract"] = (
            "달리기 자세 AI 분석" in main_source
            and "AI 분석 실행" in app_source
            and "최종 AI 모델 결과" in app_source
            and "4. 결과 / Export" in main_source
            and "육안 평가" not in main_source
            and "Skeleton Guide" not in main_source
        )
        checks["final_export_model_result_contract"] = (
            "sync_analysis_sheet(result)" in app_source
            and "AI 모델 결과 CSV" in app_source
            and "AI 모델 결과 JSON" in app_source
            and "최종 분석 결과 Excel" in app_source
        )

        mm_comparison = build_motionmetrix_comparison(
            embedded,
            {
                "overstride_mean_mm": 70.0,
                "cadence_steps_per_min": 170.0,
                "hip_rom_mean_deg": 60.0,
                "vertical_oscillation_mean_mm": 72.0,
            },
        )
        checks["motionmetrix_comparison_contract"] = (
            "overstride" in mm_comparison
            and "cadence" in mm_comparison
            and "hip_rom" in mm_comparison
            and "vertical_displacement" in mm_comparison
        )

        api_port = free_port()
        api_url = f"http://127.0.0.1:{api_port}"
        env = os.environ.copy()
        env["RUNNINGAI_API_TOKEN"] = token
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        api_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            checks["api_startup"] = wait_url(api_url + "/health")
            if checks["api_startup"]:
                info = model_info(api_url, token, timeout=180)
                http_result = analyze_debug_export(api_url, token, zip_path)
                checks["artifact_integrity"] = bool(info.get("artifact_integrity"))
                checks["embedded_api_parity"] = same_result(embedded, http_result)

                openapi = requests.get(api_url + "/openapi.json", timeout=20).json()
                raw_path = openapi.get("paths", {}).get("/api/v1/analyze/video", {})
                checks["raw_video_api_contract"] = "post" in raw_path
                auth_response = requests.post(
                    api_url + "/api/v1/analyze/video",
                    data={"patient_meta": json.dumps({"patient_id": "QA001"})},
                    files={"side_video": ("side.mp4", b"qa", "video/mp4")},
                    timeout=20,
                )
                checks["raw_video_api_auth"] = auth_response.status_code == 401
            else:
                checks["artifact_integrity"] = False
                checks["embedded_api_parity"] = False
                checks["raw_video_api_contract"] = False
                checks["raw_video_api_auth"] = False
        finally:
            api_proc.terminate()
            try:
                api_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api_proc.kill()

        streamlit_port = free_port()
        streamlit_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.headless=true",
                f"--server.port={streamlit_port}",
                "--browser.gatherUsageStats=false",
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            checks["streamlit_startup"] = wait_url(
                f"http://127.0.0.1:{streamlit_port}/_stcore/health"
            )
        finally:
            streamlit_proc.terminate()
            try:
                streamlit_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                streamlit_proc.kill()

    passed = all(checks.values())
    report = {
        "version": "integration-v1.4.0",
        "fixture": "synthetic_debug_export",
        "checks": checks,
        "raw_video_media_e2e": "run separately with a valid running MP4 before final merge",
        "pass": passed,
    }

    qa_dir = ROOT / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "INTEGRATION_QA.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Integration QA",
        "",
        f"Result: **{'PASS' if passed else 'FAIL'}**",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    lines.extend(f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in checks.items())
    (qa_dir / "INTEGRATION_QA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
