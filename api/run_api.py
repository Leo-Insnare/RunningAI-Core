from __future__ import annotations

import os
import secrets
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
TOKEN_FILE = ROOT / ".api_token"


def ensure_token() -> str:
    token = os.environ.get("RUNNINGAI_API_TOKEN")
    if token:
        return token.strip()
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            try:
                TOKEN_FILE.chmod(0o600)
            except OSError:
                pass
            return token
    token = secrets.token_urlsafe(32)
    TOKEN_FILE.write_text(token, encoding="utf-8")
    try:
        TOKEN_FILE.chmod(0o600)
    except OSError:
        pass
    return token


if __name__ == "__main__":
    ensure_token()
    print("Swagger UI: http://127.0.0.1:8000/docs")
    print("API token: RUNNINGAI_API_TOKEN or .api_token")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
