# RunningAI

Streamlit 라벨링 툴과 최종 검수 통과 모델(`final-20260825`)을 포함합니다.

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit 모델 분석은 기본적으로 embedded mode를 사용합니다.

API를 별도로 실행할 경우:

```bash
cd api
pip install -r requirements.txt
python run_api.py
```

Streamlit에서 API mode를 사용할 때는 아래 값을 설정합니다.

```text
RUNNINGAI_INFERENCE_MODE=api
RUNNINGAI_API_URL=http://127.0.0.1:8000
RUNNINGAI_API_TOKEN=<token>
```

## 검증

```bash
pip install -r api/requirements.txt
python api/scripts/verify_final_api_model.py
python scripts/validate_integration.py
```

API 규격은 `api/API_SPEC_v1.md`를 참조합니다.

## 모델 학습 및 검수 산출물

최종 모델 학습 및 독립검수 notebook은 최종 납품 패키지로 별도 제공됩니다.
