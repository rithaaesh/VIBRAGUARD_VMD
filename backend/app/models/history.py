from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from backend.app.database.session import Base


class PredictionRecord(Base):
    """Stores historical prediction records and diagnostic results."""

    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    filename = Column(String(255), nullable=True)
    machine_id = Column(String(100), default="Unknown / Not Provided")
    sampling_rate = Column(Float, default=12000.0)
    rpm = Column(String(100), default="Unknown / Not Provided")
    load_hp = Column(String(100), default="Unknown / Not Provided")

    predicted_fault = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    severity = Column(String(50), nullable=False)
    health_score = Column(Float, nullable=False)

    optimized_K = Column(Integer, default=5)
    optimized_alpha = Column(Float, default=2000.0)
    reconstruction_error = Column(Float, default=0.0)

    # JSON strings for structured breakdown
    probabilities_json = Column(Text, nullable=True)
    shap_explanation_json = Column(Text, nullable=True)
    imf_attribution_json = Column(Text, nullable=True)
    recommended_action = Column(Text, nullable=True)
