# VibraGuard System Architecture

## Overview
VibraGuard is an industrial Predictive Maintenance system incorporating Adaptive Variational Mode Decomposition (VMD) and Explainable AI (SHAP) for rotating machinery fault diagnosis.

```
Vibration Signal Input
       │
       ▼
Preprocessing & Filtering
       │
       ▼
Adaptive VMD Parameter Selection (K, alpha)
       │
       ▼
IMF Signal Decomposition & Feature Extraction
       │
       ▼
ML Classifier (Random Forest / SVM / Gradient Boosting)
       │
       ▼
XAI Diagnostics (SHAP Feature / IMF Attribution)
       │
       ▼
Industrial Health Dashboard
```

## Backend Services Layer
- `backend/app/api/`: REST Endpoints (Signal upload, VMD analysis, Inference, Training, History)
- `backend/app/core/`: Configuration, Security, Environment settings
- `backend/app/services/`: Core business logic adapters
- `backend/app/models/`: SQLAlchemy ORM models for database persistence
- `backend/app/schemas/`: Pydantic schemas for request/response validation
