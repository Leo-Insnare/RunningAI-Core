# Parity Validation

기존 v0.16 base/Strike parity 검증 결과를 유지하고, API v1.4.0에서 최종 Overstride 보정층과 Strike review threshold 0.41을 반영했습니다. 최종 모델 parity는 `FINAL_MODEL_PARITY.md`를 참조합니다.

## Debug Export

기존 개발용 Debug Export 20건을 v0.16 개발 artifact와 비교했습니다. 환자 원본 데이터는 저장소에 포함하지 않습니다.

| 항목 | 결과 |
|---|---:|
| Debug Export packages | 20 |
| Strike event sequences | 294 |
| Strike sequence shape | 21 x 17 |
| Sequence max absolute difference | 0.0 |
| Sequence NaN pattern mismatch | 0 |
| Strike rule-class mismatch | 0 |
| Foot/event count mismatch | 0 |
| Cadence patients | 20 |
| Cadence max absolute difference | 2.842170943040401e-14 spm |
| Canonical pass-through max absolute difference | 0.0 |

Strike sequence는 `strike_pose_sequence_dataset_v0_16.npz`, cadence는 `cadence_same_foot_final_audit_v0_16.csv` 기준으로 비교했습니다.

## API contract

- Direct Overstride 요청의 `height_cm`, `running_speed_kmh`를 `patient_meta`에서 모델 입력으로 연결
- Strike 좌우 local output과 v0.16 최종 patient-consensus output을 분리
- OpenAPI Bearer security scheme 적용
- 모델 artifact SHA-256 고정

## Runtime

- `scikit-learn==1.6.1`
- `joblib==1.4.2`
- hardening 확인 환경: Python 3.12.13 / PyTorch 2.11.0+cu128

패키지 테스트 결과는 `14 passed`입니다. 별도 Debug Export 1건을 reference sequence와 다시 비교했으며 14개 event의 sequence 최대 차이는 `0.0`, NaN/rule mismatch는 `0`이었습니다.

개별 Debug Export parity는 `scripts/validate_parity.py`로 확인할 수 있습니다.
