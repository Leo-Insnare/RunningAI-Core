from __future__ import annotations

from typing import BinaryIO

import requests


def analyze_debug_export(api_url: str, token: str, file_obj: BinaryIO, filename: str = "debug_export.zip") -> dict:
    response = requests.post(
        api_url.rstrip("/") + "/api/v1/analyze/debug-export",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, file_obj, "application/zip")},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def build_api_request(api_url: str, token: str, file_obj: BinaryIO, filename: str = "debug_export.zip") -> dict:
    response = requests.post(
        api_url.rstrip("/") + "/api/v1/adapter/debug-export",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, file_obj, "application/zip")},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()
