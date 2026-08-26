# Final Model Parity

Result: **PASS**

| Check | Result |
|---|---:|
| Model version | final-20260825 |
| Independent patients | 14 |
| Overstride API inference vs frozen final prediction max abs diff | 2.842e-14 mm |
| Final Overstride MAE | 19.15 mm |
| Overstride criterion | ≤ 20.00 mm |
| Final Strike accuracy | 85.71% |
| Strike criterion | ≥ 80.00% |
| Strike review threshold | 0.41 |

최종 Overstride API inference artifact는 동결된 최종 Ridge 모델의 affine inference representation이며, 최종 검수 예측과 machine precision 범위에서 일치합니다. v0.16 Overstride ensemble은 base prediction 생성에 계속 사용합니다.
