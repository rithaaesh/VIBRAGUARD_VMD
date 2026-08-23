import os
import json
import joblib
import io
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session

from backend.app.database.session import get_db, engine, Base
from backend.app.models.history import PredictionRecord
from backend.app.schemas.predict import (
    PredictRequest,
    PredictResponse,
    PredictionHistoryItem,
    FeatureContributionSchema,
)

from ml.preprocessing.dataset_adapter import GenericCSVAdapter, SignalMetadata
from ml.preprocessing.signal_preprocessor import SignalPreprocessor
from ml.vmd.vmd_engine import VMDEngine
from ml.optimization.vmd_optimizer import VMDOptimizer
from ml.features.feature_extractor import FeatureExtractor
from ml.explainability.shap_explainer import SHAPExplainer

# Initialize tables
Base.metadata.create_all(bind=engine)

router = APIRouter(prefix="/predict", tags=["Fault Prediction & XAI"])

preprocessor = SignalPreprocessor()
feature_extractor = FeatureExtractor()


def _compute_health_score_and_severity(
    predicted_fault: str, confidence: float, rms: float, kurtosis: float
) -> tuple[float, str, str]:
    """
    Engineering heuristic for calculating Machine Health Score (0 to 100) and Severity Level.
    State plainly this is an engineering heuristic, not a certified metric.
    """
    health = 100.0

    # Severity Deductions
    fault_lower = predicted_fault.lower()
    if "normal" in fault_lower or "healthy" in fault_lower:
        severity = "HEALTHY"
        action = "System operates within normal vibration tolerances. Routine inspection recommended per standard schedule."
        health -= (1.0 - confidence) * 10.0
    elif "inner" in fault_lower:
        severity = "HIGH SEVERITY"
        action = "Inner race bearing defect detected. Schedule immediate maintenance window to inspect bearing assembly."
        health -= 45.0 + (confidence * 15.0)
    elif "outer" in fault_lower:
        severity = "MEDIUM SEVERITY"
        action = "Outer race bearing defect detected. Monitor vibration trend closely; plan replacement during next service window."
        health -= 35.0 + (confidence * 15.0)
    elif "ball" in fault_lower:
        severity = "MEDIUM SEVERITY"
        action = "Rolling element (ball) defect detected. Check lubrication and align shaft coupling."
        health -= 35.0 + (confidence * 10.0)
    else:
        severity = "WARNING"
        action = "Anomalous vibration pattern detected. Conduct manual diagnostic check."
        health -= 25.0

    # RMS & Kurtosis Penalties
    if kurtosis > 4.0:
        health -= min(15.0, (kurtosis - 4.0) * 3.0)

    health_score = max(0.0, min(100.0, float(health)))
    return health_score, severity, action


def _execute_end_to_end_pipeline(
    signal_array: np.ndarray,
    metadata: SignalMetadata,
    adaptive_vmd: bool = True,
    db: Optional[Session] = None,
) -> PredictResponse:
    """Core end-to-end processing pipeline."""
    # 1. Preprocess signal
    fs = metadata.sampling_rate
    prep_res = preprocessor.process(signal_array, fs=fs, detrend=True)
    clean_signal = prep_res["processed_signal"]

    # 2. Adaptive / Fixed VMD
    if adaptive_vmd:
        optimizer = VMDOptimizer()
        opt_res = optimizer.optimize_deterministic(clean_signal[:2048], fs=fs, k_range=(3, 6))
        K = opt_res["optimized_K"]
        alpha = opt_res["optimized_alpha"]
        vmd_engine = VMDEngine(K=K, alpha=alpha)
        vmd_decomp = vmd_engine.decompose(clean_signal, fs=fs)
        recon_err = opt_res["optimized_decomposition"]["reconstruction_error"]
    else:
        K = 5
        alpha = 2000.0
        vmd_engine = VMDEngine(K=K, alpha=alpha)
        vmd_decomp = vmd_engine.decompose(clean_signal, fs=fs)
        recon_err = vmd_decomp["reconstruction_error"]

    # 3. Extract Features
    feat_dict = feature_extractor.extract_feature_dict(clean_signal, fs=fs, imf_matrix=vmd_decomp["raw_imfs"])

    # 4. Load Models & Predict
    model_dir = "models"
    if not os.path.exists(os.path.join(model_dir, "model.joblib")):
        # Auto-train default model if missing
        from ml.training.trainer import ModelTrainer
        ModelTrainer(model_dir=model_dir).train_and_evaluate("Random Forest")

    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

    with open(os.path.join(model_dir, "feature_schema.json"), "r") as f:
        schema = json.load(f)

    feat_names = schema["feature_names"]
    vector = pd.DataFrame(
        [[float(feat_dict.get(fn, 0.0)) for fn in feat_names]],
        columns=feat_names,
    )
    scaled_vector = scaler.transform(vector)

    pred_enc = model.predict(scaled_vector)[0]
    predicted_fault = str(label_encoder.inverse_transform([pred_enc])[0])

    probs = model.predict_proba(scaled_vector)[0]
    confidence = float(np.max(probs))

    classes = label_encoder.classes_.tolist()
    class_probabilities = {cls: float(probs[i]) for i, cls in enumerate(classes)}

    # 5. Compute Health Score & Severity
    raw_rms = feat_dict.get("raw_rms", 0.0)
    raw_kurtosis = feat_dict.get("raw_kurtosis", 0.0)
    health_score, severity, action = _compute_health_score_and_severity(
        predicted_fault, confidence, raw_rms, raw_kurtosis
    )

    # 6. SHAP Explainability
    explainer = SHAPExplainer(model_dir=model_dir)
    shap_res = explainer.explain_sample(feat_dict, top_k=6)

    top_feat_schemas = [FeatureContributionSchema(**f) for f in shap_res["top_features"]]

    # 7. Database Persistence
    pred_id = None
    if db is not None:
        rec = PredictionRecord(
            timestamp=datetime.utcnow(),
            filename=getattr(metadata, "filename", "inline_signal.csv"),
            machine_id=str(metadata.machine_id),
            sampling_rate=fs,
            rpm=str(metadata.rpm),
            load_hp=str(metadata.load_hp),
            predicted_fault=predicted_fault,
            confidence=confidence,
            severity=severity,
            health_score=health_score,
            optimized_K=K,
            optimized_alpha=alpha,
            reconstruction_error=recon_err,
            probabilities_json=json.dumps(class_probabilities),
            shap_explanation_json=json.dumps(shap_res["top_features"]),
            imf_attribution_json=json.dumps(shap_res["imf_attribution"]),
            recommended_action=action,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        pred_id = rec.id

    return PredictResponse(
        prediction_id=pred_id,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        predicted_fault=predicted_fault,
        confidence=confidence,
        severity=severity,
        health_score=health_score,
        health_score_disclaimer="Machine Health Score is an engineering heuristic combining fault severity, model confidence, and vibration amplitude metrics.",
        class_probabilities=class_probabilities,
        optimized_K=K,
        optimized_alpha=alpha,
        reconstruction_error=recon_err,
        top_features=top_feat_schemas,
        imf_attribution=shap_res["imf_attribution"],
        recommended_action=action,
        xai_disclaimer=shap_res["disclaimer"],
    )


@router.post("", response_model=PredictResponse)
async def predict_fault_payload(req: PredictRequest, db: Session = Depends(get_db)):
    """
    Executes end-to-end fault inference + adaptive VMD + SHAP explainability on signal payload.
    """
    if not req.signal or len(req.signal) < 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Signal array must contain at least 128 numeric samples.",
        )

    try:
        raw_signal = np.array(req.signal, dtype=np.float64)
        metadata = SignalMetadata(
            sampling_rate=req.sampling_rate or 12000.0,
            rpm=req.rpm,
            load_hp=req.load_hp,
            machine_id=req.machine_id or "PUMP-101",
            dataset_source="API Predict JSON",
        )
        setattr(metadata, "filename", req.filename or "stream_signal.csv")

        return _execute_end_to_end_pipeline(
            signal_array=raw_signal,
            metadata=metadata,
            adaptive_vmd=req.adaptive_vmd if req.adaptive_vmd is not None else True,
            db=db,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline error: {str(e)}",
        )


@router.post("/file", response_model=PredictResponse)
async def predict_fault_file(
    file: UploadFile = File(...),
    sampling_rate: Optional[float] = Form(12000.0),
    rpm: Optional[float] = Form(None),
    load_hp: Optional[float] = Form(None),
    machine_id: Optional[str] = Form("PUMP-101"),
    adaptive_vmd: bool = Form(True),
    db: Session = Depends(get_db),
):
    """
    Upload CSV signal file and execute end-to-end fault prediction & XAI diagnostics.
    """
    if not file.filename or not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only CSV or TXT files are accepted.",
        )

    try:
        content = await file.read()
        csv_adapter = GenericCSVAdapter()
        meta_override = {
            "sampling_rate": sampling_rate,
            "rpm": rpm,
            "load_hp": load_hp,
            "machine_id": machine_id,
        }
        signal_array, metadata = csv_adapter.load_signal(io.BytesIO(content), meta_override)
        setattr(metadata, "filename", file.filename)

        return _execute_end_to_end_pipeline(
            signal_array=signal_array,
            metadata=metadata,
            adaptive_vmd=adaptive_vmd,
            db=db,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File inference error: {str(e)}",
        )


@router.get("/history", response_model=List[PredictionHistoryItem])
async def get_prediction_history(limit: int = 20, db: Session = Depends(get_db)):
    """
    Queries past prediction records stored in SQLite database.
    """
    records = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.timestamp.desc())
        .limit(limit)
        .all()
    )

    history = [
        PredictionHistoryItem(
            id=rec.id,
            timestamp=rec.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            filename=rec.filename or "signal.csv",
            machine_id=rec.machine_id or "Unknown",
            predicted_fault=rec.predicted_fault,
            confidence=rec.confidence,
            severity=rec.severity,
            health_score=rec.health_score,
            optimized_K=rec.optimized_K or 5,
            recommended_action=rec.recommended_action or "Routine inspection",
        )
        for rec in records
    ]
    return history
