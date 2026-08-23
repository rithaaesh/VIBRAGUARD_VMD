from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class PredictRequest(BaseModel):
    signal: List[float] = Field(..., description="1D raw vibration array")
    sampling_rate: Optional[float] = 12000.0
    rpm: Optional[float] = None
    load_hp: Optional[float] = None
    machine_id: Optional[str] = "PUMP-101"
    filename: Optional[str] = "live_sensor_stream.csv"
    adaptive_vmd: Optional[bool] = True


class FeatureContributionSchema(BaseModel):
    feature_name: str
    shap_value: float
    importance: float
    feature_value: float
    impact_direction: str
    explanation_text: str


class PredictResponse(BaseModel):
    prediction_id: Optional[int] = None
    timestamp: str
    predicted_fault: str
    confidence: float
    severity: str
    health_score: float
    health_score_disclaimer: str
    class_probabilities: Dict[str, float]
    optimized_K: int
    optimized_alpha: float
    reconstruction_error: float
    top_features: List[FeatureContributionSchema]
    imf_attribution: Dict[str, float]
    recommended_action: str
    xai_disclaimer: str


class PredictionHistoryItem(BaseModel):
    id: int
    timestamp: str
    filename: str
    machine_id: str
    predicted_fault: str
    confidence: float
    severity: str
    health_score: float
    optimized_K: int
    recommended_action: str
