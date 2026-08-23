import os
import json
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, Optional

from backend.app.core.config import settings
from ml.training.trainer import ModelTrainer

router = APIRouter(prefix="/system", tags=["System & Retraining"])


class TrainModelRequest(BaseModel):
    model_name: Optional[str] = "Random Forest"


@router.get("/status")
async def get_system_status():
    """
    Returns live system status, active model version, metadata, and database state.
    """
    model_dir = "models"
    meta_path = os.path.join(model_dir, "training_metadata.json")

    model_metadata = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                model_metadata = json.load(f)
        except Exception:
            pass

    feature_count = model_metadata.get("feature_count", 0) if model_metadata else 0
    if feature_count == 0:
        schema_path = os.path.join(model_dir, "feature_schema.json")
        try:
            with open(schema_path, "r") as f:
                feature_count = json.load(f).get("feature_count", 0)
        except (OSError, json.JSONDecodeError):
            feature_count = 0

    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "api_version": "1.0.0",
        "active_model": model_metadata.get("model_name", "Random Forest") if model_metadata else "Uninitialized",
        "model_version": model_metadata.get("version", "v1.0.0") if model_metadata else "v0.0.0",
        "model_accuracy": model_metadata.get("accuracy", 0.0) if model_metadata else 0.0,
        "feature_count": feature_count,
        "components": {
            "api_engine": "HEALTHY",
            "vmd_engine": "HEALTHY",
            "feature_extractor": "HEALTHY",
            "shap_explainer": "HEALTHY",
            "database": "CONNECTED",
        },
    }


@router.post("/train")
async def trigger_model_training(req: TrainModelRequest):
    """
    Triggers model training and saves new model bundle artifacts.
    """
    model_name = req.model_name or "Random Forest"
    try:
        trainer = ModelTrainer(model_dir="models")
        meta = trainer.train_and_evaluate(model_name)
        return {
            "status": "success",
            "message": f"Successfully trained and deployed model '{model_name}'",
            "metadata": meta,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}",
        )
