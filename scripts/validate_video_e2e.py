from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests
from starlette.datastructures import UploadFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app.main import analyze
from api.app.video_analysis import build_analyze_request_from_videos, build_motionmetrix_comparison
from src.api_integration import analyze_video_api


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return int(port)


def wait_url(url: str, timeout_sec: int = 90) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


async def direct_result(
    side_path: Path,
    rear_path: Path | None,
    patient_meta: dict,
    motionmetrix: dict,
    session_id: str,
    start: float,
    duration: float,
) -> dict:
    side_file = side_path.open("rb")
    rear_file = rear_path.open("rb") if rear_path else None
    try:
        side_upload = UploadFile(file=side_file, filename=side_path.name)
        rear_upload = UploadFile(file=rear_file, filename=rear_path.name) if rear_file else None
        built = await build_analyze_request_from_videos(
            side_video=side_upload,
            rear_video=rear_upload,
            session_id=session_id,
            patient_meta_raw=patient_meta,
            motionmetrix=motionmetrix,
            analysis_start_sec=start,
            analysis_duration_sec=duration,
        )
        response = analyze(built.request)
        payload = response.model_dump()
        payload["motionmetrix_comparison"] = build_motionmetrix_comparison(payload, motionmetrix)
        payload["motionmetrix_reference"] = motionmetrix
        payload["source"] = built.source.model_dump()
        return payload
    finally:
        side_file.close()
        if rear_file:
            rear_file.close()


def num_equal(a, b, tol=1e-8) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return a == b


def compare(direct: dict, http: dict) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for path in [
        ("overstride", "prediction_mm"),
        ("overstride", "prediction_std_mm"),
        ("strike_type", "final_confidence"),
    ]:
        a = direct.get(path[0], {}).get(path[1])
        b = http.get(path[0], {}).get(path[1])
        if not num_equal(a, b):
            mismatches.append(".".join(path))
    for path in [
        ("overstride", "status"),
        ("strike_type", "status"),
        ("strike_type", "final_class"),
        ("strike_type", "review_required"),
    ]:
        if direct.get(path[0], {}).get(path[1]) != http.get(path[0], {}).get(path[1]):
            mismatches.append(".".join(path))

    dp = direct.get("posture_metrics") or {}
    hp = http.get("posture_metrics") or {}
    for key in sorted(set(dp) | set(hp)):
        if not num_equal(dp.get(key), hp.get(key), tol=1e-6):
            mismatches.append(f"posture_metrics.{key}")

    if direct.get("motionmetrix_comparison") != http.get("motionmetrix_comparison"):
        mismatches.append("motionmetrix_comparison")
    return not mismatches, mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("side_video")
    parser.add_argument("--rear-video", default="")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--height-cm", type=float, default=None)
    parser.add_argument("--weight-kg", type=float, default=None)
    parser.add_argument("--speed-kmh", type=float, default=None)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--motionmetrix-json", default="")
    args = parser.parse_args()

    side = Path(args.side_video).resolve()
    rear = Path(args.rear_video).resolve() if args.rear_video else None
    if not side.exists():
        raise SystemExit(f"Side video not found: {side}")
    if rear and not rear.exists():
        raise SystemExit(f"Rear video not found: {rear}")

    patient_meta = {
        "patient_id": args.patient_id,
        "height_cm": args.height_cm,
        "weight_kg": args.weight_kg,
        "running_speed_kmh": args.speed_kmh,
    }
    motionmetrix = {}
    if args.motionmetrix_json:
        motionmetrix = json.loads(Path(args.motionmetrix_json).read_text(encoding="utf-8"))

    token = "video-e2e-test-token"
    port = free_port()
    api_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["RUNNINGAI_API_TOKEN"] = token
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
    )
    try:
        if not wait_url(api_url + "/health"):
            raise RuntimeError("API startup failed.")
        session_id = "VIDEO_E2E_QA"
        direct = asyncio.run(
            direct_result(
                side,
                rear,
                patient_meta,
                motionmetrix,
                session_id,
                args.start,
                args.duration,
            )
        )
        http = analyze_video_api(
            api_url,
            token,
            side,
            patient_meta,
            rear_video_path=rear,
            motionmetrix=motionmetrix,
            session_id=session_id,
            analysis_start_sec=args.start,
            analysis_duration_sec=args.duration,
            timeout=1800,
        )
        passed, mismatches = compare(direct, http)
        report = {
            "version": "integration-v1.4.0",
            "side_video": side.name,
            "rear_video": rear.name if rear else None,
            "analysis_start_sec": args.start,
            "analysis_duration_sec": args.duration,
            "direct_overstride_mm": direct.get("overstride", {}).get("prediction_mm"),
            "http_overstride_mm": http.get("overstride", {}).get("prediction_mm"),
            "direct_strike": direct.get("strike_type", {}).get("final_class"),
            "http_strike": http.get("strike_type", {}).get("final_class"),
            "mismatches": mismatches,
            "pass": passed,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if passed else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
