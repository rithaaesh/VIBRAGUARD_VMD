# VibraGuard API Documentation

## System Endpoints
- `GET /api/health`: System health and version indicator.

## Signal Processing & ML Endpoints
- `POST /api/v1/signal/upload`: Process uploaded CSV signal.
- `POST /api/v1/signal/process`: Process a raw JSON signal array.
- `POST /api/v1/vmd/decompose`: Execute fixed or adaptive VMD decomposition.
- `POST /api/v1/vmd/optimize`: Run deterministic K/alpha search and return the fitness trajectory.
- `POST /api/v1/predict`: End-to-end fault inference and SHAP explainability.
- `POST /api/v1/predict/file`: Run inference from an uploaded CSV signal.
- `GET /api/v1/predict/history`: Retrieve historical prediction records.

## Operations
- `GET /api/v1/system/status`: Report API, database, model, VMD, and SHAP status.
- `POST /api/v1/system/train`: Train and deploy a supported model.

Invalid signal/file input returns an actionable `400` or `422` response. Server-side
processing errors are returned as structured FastAPI error responses rather than raw traces.
