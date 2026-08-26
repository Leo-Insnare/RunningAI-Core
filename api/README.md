# RunningAI API v1.4.0

최종 검수 통과 모델(`final-20260825`) 통합본입니다. Python 3.12 환경을 권장합니다. Debug Export ZIP, raw video, 직접 JSON 요청을 지원합니다.

## 실행

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
python run_api.py
```

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- `/api/v1/*`: Bearer Token 필요

## 주요 API

- `GET /health`
- `GET /api/v1/model/info`
- `POST /api/v1/predict/overstride`
- `POST /api/v1/predict/strike-type`
- `POST /api/v1/analyze`
- `POST /api/v1/adapter/debug-export`
- `POST /api/v1/analyze/debug-export`
- `POST /api/v1/analyze/video`

Overstride는 v0.16 30-pair base ensemble과 최종 Ridge 보정층의 동등 inference export를 사용합니다. Strike 최종 판정값은 `final_class`이며 review threshold는 `0.41`입니다.

상세 규격은 `API_SPEC_v1.md`, 검증 근거는 `PARITY_VALIDATION.md`, `FINAL_MODEL_PARITY.md`를 참조합니다.
