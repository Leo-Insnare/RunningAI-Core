# RunningAI REST API v1.4.0

## Authentication

`/health`를 제외한 `/api/v1/*` 요청은 Bearer Token을 사용합니다. Swagger의 `Authorize`에서 동일한 token을 입력할 수 있습니다.

```text
Authorization: Bearer <token>
```

## Debug Export

### POST `/api/v1/adapter/debug-export`

환자 1명의 Debug Export ZIP을 `multipart/form-data`의 `file`로 받습니다. 응답은 `/api/v1/analyze`에 그대로 사용할 수 있는 `AnalyzeRequest`입니다.

### POST `/api/v1/analyze/debug-export`

Debug Export ZIP을 변환한 뒤 최종 검수 통과 모델(`final-20260825`)을 실행합니다.

추론 입력에 사용하는 파일:

- `session/session_meta.json`
- `processed/side_running/side_running_all_frame_metrics.csv`
- `processed/side_running/side_running_gait_events.csv`
- `processed/side_running/side_running_clip_summary.csv`
- `processed/rear_running/rear_running_all_frame_metrics.csv`
- `processed/rear_running/rear_running_gait_events.csv`
- `processed/rear_running/rear_running_clip_summary.csv`

`session/motionmetrix_values.json`은 추론 입력에 사용하지 않습니다.

## Overstride

최종 예측은 v0.16 30-pair Overstride ensemble의 base prediction에 최종 Ridge 보정층을 적용합니다. Debug Export/raw video adapter가 아래 보정 feature를 자동 생성합니다. 직접 JSON 요청에서는 동일 feature가 필요합니다.

- `knee_landing_event_median_deg`
- `thigh_flexion_recalc_deg`
- `hip_rom_recalc_deg`

필수 보정 feature가 생성되지 않으면 임의 보간하지 않고 `unmeasurable`을 반환합니다.


`patient_meta.height_cm`, `patient_meta.running_speed_kmh`는 모델 입력에 자동 반영됩니다. 동일 키가 `features`에도 있는 경우 `patient_meta` 값을 사용합니다.

주요 입력 feature:

- `os_clip_selected_mm`
- `os_event_abs_mean_mm`
- `os_event_abs_iqr_mm`
- `os_lr_abs_diff_mm`
- `foot_angle_abs_median_deg`
- `shank_angle_abs_median_deg`
- `knee_landing_event_median_deg`
- `running_speed_kmh`
- `height_cm`
- `side_pose_rate`

`os_clip_selected_mm`, `os_event_abs_mean_mm`의 square/log transform은 서버에서 생성합니다.

## Strike Type

입력 단위는 gait event별 `21 x 17` sequence입니다. 시간축은 `-250 ms ~ +250 ms`, 25 ms 간격입니다.

Channel order:

1. `heel_x_rel_norm`
2. `toe_x_rel_norm`
3. `ankle_x_rel_norm`
4. `heel_ground_gap_norm`
5. `toe_ground_gap_norm`
6. `ankle_ground_gap_norm`
7. `heel_toe_y_diff_norm`
8. `foot_angle_canonical_deg`
9. `foot_angle_delta_ic_deg`
10. `shank_angle_abs_deg`
11. `shank_angle_delta_ic_deg`
12. `knee_flexion_deg`
13. `knee_flexion_delta_ic_deg`
14. `heel_gap_velocity_norm_s`
15. `toe_gap_velocity_norm_s`
16. `heel_toe_velocity_norm_s`
17. `foot_angle_velocity_deg_s`

응답 구분:

- `feet[].prediction`: 좌/우 발별 probability의 argmax. 참고값
- `feet[].local_probabilities`: 좌/우 발별 probability
- `patient_anchor_class`: 좌우 probability를 평균한 patient consensus의 rescue 적용 전 class
- `final_class`: v0.16 frozen 최종 판정값
- `final_confidence`: 최종 판정 confidence
- `review_required`: 최종 판정 기준 검토 필요 여부

최종 Strike 성능 검증 대상은 `final_class`이며 review confidence threshold는 `0.41`입니다. 좌우 local output은 별도 참고값이며 최종 성능 지표를 대체하지 않습니다.

Rescue rule:

```text
patient_anchor_class == forefoot and mean P(heel) >= 0.65 -> final_class = heel
```

Critical channel이 환자 전체에서 완전히 소실된 경우 `unmeasurable`을 반환합니다.

## Posture metrics

Debug Export adapter가 계산하는 값:

- cadence: same-foot IC cycle
- contact time: IC→toe-off + 1 frame
- forward lean: frame mean
- max thigh flexion: 좌우 95 percentile 평균
- max thigh extension: 좌우 5 percentile extension magnitude 평균
- knee flexion at touchdown: gait-event mean
- shank angle at touchdown: gait-event median
- pelvic drop / hip hike difference: rear IC→toe-off stance descriptive value

Pelvic Drop은 descriptive output이며 classifier를 포함하지 않습니다.

## Consistency

`/api/v1/analyze`, `/api/v1/analyze/debug-export`, raw video 분석은 동일한 최종 frozen engine을 사용합니다. base/Strike artifact와 최종 Overstride inference artifact의 SHA-256이 고정값과 다르면 서버 초기화 단계에서 오류가 발생합니다.
