from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from .config import API_VERSION, MODEL_VERSION, load_status
from .debug_export_adapter import DebugExportAdapter
from .model_engine import FrozenModelEngine
from .schemas import (
    AnalysisSource,
    AnalyzeRequest,
    AnalyzeResponse,
    ModelInfoResponse,
    OverstrideRequest,
    OverstrideResponse,
    QualityInfo,
    StrikeRequest,
    StrikeResponse,
)
from .security import require_token
from .video_analysis import (
    build_analyze_request_from_videos,
    build_motionmetrix_comparison,
    parse_json_object,
)

app = FastAPI(title="RunningAI API", version="1.4.0")


@lru_cache(maxsize=1)
def get_engine() -> FrozenModelEngine:
    return FrozenModelEngine()


def _overstride_features(req: OverstrideRequest) -> dict[str, float | None]:
    features = dict(req.features)
    if req.patient_meta.height_cm is not None:
        features["height_cm"] = req.patient_meta.height_cm
    if req.patient_meta.running_speed_kmh is not None:
        features["running_speed_kmh"] = req.patient_meta.running_speed_kmh
    return features


def _final_quality(req_quality: QualityInfo, overstride: OverstrideResponse, strike: StrikeResponse) -> QualityInfo:
    reasons = list(req_quality.unmeasurable_reasons)
    if overstride.status == "unmeasurable" and overstride.reason:
        reasons.append(f"overstride:{overstride.reason}")
    if strike.status == "unmeasurable" and strike.reason:
        reasons.append(f"strike:{strike.reason}")
    reasons = list(dict.fromkeys(reasons))

    if overstride.status == "unmeasurable" or strike.status == "unmeasurable":
        status = "unmeasurable"
    elif overstride.review_required or strike.review_required or req_quality.quality_tier == "low":
        status = "review_required"
    else:
        status = "completed"

    return req_quality.model_copy(
        update={
            "status": status,
            "unmeasurable_reasons": reasons,
        }
    )


@app.get("/health")
def health():
    return {"status": "ok", "api_version": API_VERSION, "model_version": MODEL_VERSION}


@app.get(
    "/api/v1/model/info",
    response_model=ModelInfoResponse,
    dependencies=[Depends(require_token)],
)
def model_info():
    engine = get_engine()
    status = load_status()
    return ModelInfoResponse(
        api_version=API_VERSION,
        model_version=MODEL_VERSION,
        stage="final-independent-completed" if status.get("final_independent_test_completed") else "frozen-pre-final-independent",
        supported_metrics=[
            "overstride",
            "strike_type",
            "cadence",
            "contact_time",
            "forward_lean",
            "max_thigh_flexion",
            "max_thigh_extension",
            "hip_rom",
            "knee_flexion_touchdown",
            "max_knee_flexion_swing",
            "knee_rom",
            "vertical_displacement",
            "pelvic_drop",
            "hip_hike_difference",
            "shank_angle_touchdown",
        ],
        final_independent_test_completed=bool(status.get("final_independent_test_completed", False)),
        final_independent_targets_met=status.get("final_independent_targets_met"),
        artifact_integrity=engine.status.artifact_integrity,
        strike_aggregation="bilateral patient-consensus",
        strike_rescue_rule="anchor=forefoot and rescue P(heel)>=0.65",
        notes=[
            "feet[].prediction is a per-foot auxiliary output.",
            "Final Strike performance gate applies to final_class with review threshold 0.41.",
            "Raw video, Debug Export, and direct JSON requests use the same final frozen model engine.",
        ],
    )


@app.post(
    "/api/v1/predict/overstride",
    response_model=OverstrideResponse,
    dependencies=[Depends(require_token)],
)
def predict_overstride(req: OverstrideRequest):
    try:
        raw = get_engine().predict_overstride(_overstride_features(req))
    except Exception as e:
        raise HTTPException(status_code=422, detail={"code": "OVERSTRIDE_INPUT_ERROR", "message": str(e)}) from e
    quality = req.quality_tier
    review = raw["status"] != "completed" or (quality is not None and quality == "low")
    return OverstrideResponse(
        session_id=req.session_id,
        patient_id=req.patient_meta.patient_id,
        prediction_mm=raw["prediction_mm"],
        prediction_std_mm=raw["prediction_std_mm"],
        model_pair_count=raw["model_pair_count"],
        quality_tier=quality,
        review_required=review,
        status=raw["status"],
        reason=raw["reason"],
        model_version=MODEL_VERSION,
    )


@app.post(
    "/api/v1/predict/strike-type",
    response_model=StrikeResponse,
    dependencies=[Depends(require_token)],
)
def predict_strike(req: StrikeRequest):
    try:
        raw = get_engine().predict_strike([x.model_dump() for x in req.feet])
    except Exception as e:
        raise HTTPException(status_code=422, detail={"code": "STRIKE_INPUT_ERROR", "message": str(e)}) from e
    return StrikeResponse(
        session_id=req.session_id,
        patient_id=req.patient_meta.patient_id,
        status=raw["status"],
        reason=raw["reason"],
        feet=raw["feet"],
        patient_anchor_class=raw["patient_anchor_class"],
        patient_anchor_confidence=raw["patient_anchor_confidence"],
        final_class=raw["final_class"],
        final_confidence=raw["final_confidence"],
        review_required=raw["review_required"],
        rescue_p_heel=raw["rescue_p_heel"],
        rescue_activated=raw["rescue_activated"],
        model_version=MODEL_VERSION,
    )


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_token)],
)
def analyze(req: AnalyzeRequest):
    overstride = predict_overstride(OverstrideRequest(
        session_id=req.session_id,
        patient_meta=req.patient_meta,
        features=req.overstride_features,
        quality_tier=req.quality.quality_tier,
    ))
    strike = predict_strike(StrikeRequest(
        session_id=req.session_id,
        patient_meta=req.patient_meta,
        feet=req.strike_feet,
    ))
    info = model_info()
    quality = _final_quality(req.quality, overstride, strike)
    return AnalyzeResponse(
        session_id=req.session_id,
        patient_id=req.patient_meta.patient_id,
        overstride=overstride,
        strike_type=strike,
        posture_metrics=req.posture_metrics,
        quality=quality,
        model_info=info.model_dump(),
        source=AnalysisSource(input_type="json_features"),
    )


@app.post(
    "/api/v1/adapter/debug-export",
    response_model=AnalyzeRequest,
    dependencies=[Depends(require_token)],
)
async def adapt_debug_export(file: UploadFile = File(...)):
    try:
        result = DebugExportAdapter.from_bytes(await file.read()).build()
        return result.request
    except Exception as e:
        raise HTTPException(status_code=422, detail={"code": "DEBUG_EXPORT_ERROR", "message": str(e)}) from e


@app.post(
    "/api/v1/analyze/debug-export",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_token)],
)
async def analyze_debug_export(file: UploadFile = File(...)):
    try:
        result = DebugExportAdapter.from_bytes(await file.read()).build()
    except Exception as e:
        raise HTTPException(status_code=422, detail={"code": "DEBUG_EXPORT_ERROR", "message": str(e)}) from e
    response = analyze(result.request)
    return response.model_copy(update={"source": AnalysisSource(input_type="debug_export")})


@app.post(
    "/api/v1/analyze/video",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_token)],
)
async def analyze_video(
    side_video: UploadFile = File(..., description="Side-view running MP4, maximum 60 seconds / 300 MB"),
    rear_video: UploadFile | None = File(None, description="Optional rear-view running MP4"),
    patient_meta: str = Form(..., description="JSON object. patient_id is required."),
    motionmetrix: str | None = Form(None, description="Optional MotionMetrix JSON object"),
    session_id: str | None = Form(None),
    analysis_start_sec: float = Form(0.0, ge=0.0),
    analysis_duration_sec: float = Form(60.0, ge=0.5, le=60.0),
):
    try:
        patient_meta_raw = parse_json_object(patient_meta, "patient_meta", required=True)
        motionmetrix_raw = parse_json_object(motionmetrix, "motionmetrix", required=False)
        built = await build_analyze_request_from_videos(
            side_video=side_video,
            rear_video=rear_video,
            session_id=session_id,
            patient_meta_raw=patient_meta_raw,
            motionmetrix=motionmetrix_raw,
            analysis_start_sec=float(analysis_start_sec),
            analysis_duration_sec=float(analysis_duration_sec),
        )
        response = analyze(built.request)
        payload = response.model_dump()
        comparisons = build_motionmetrix_comparison(payload, built.motionmetrix)
        payload.update(
            {
                "motionmetrix_comparison": comparisons,
                "motionmetrix_reference": built.motionmetrix,
                "source": built.source.model_dump(),
            }
        )
        return AnalyzeResponse.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "VIDEO_METADATA_ERROR", "message": str(e)},
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "VIDEO_INPUT_ERROR", "message": str(e)},
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "VIDEO_ANALYSIS_ERROR", "message": str(e)},
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "VIDEO_UNMEASURABLE", "message": str(e)},
        ) from e
