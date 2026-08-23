# VibraGuard 🛡️
### Adaptive VMD-Based Predictive Maintenance and Explainable Fault Diagnosis System

VibraGuard is an industrial-grade predictive maintenance web application and signal analysis platform. It processes vibration signals from rotating machinery using Variational Mode Decomposition (VMD), extracts time/frequency condition indicators, classifies faults using machine learning models, and provides explainable AI (XAI) diagnostics using SHAP.

---

## 🏗️ Architecture

- **Backend**: Python 3.11, FastAPI, Uvicorn, SQLAlchemy, SQLite, Pydantic v2
- **Signal & ML**: NumPy, SciPy, pandas, scikit-learn, vmdpy, SHAP
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide React, Framer Motion, Plotly.js

---

## ⚡ Quick Start

### 1. Backend Setup
```bash
# From project root
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# Start Backend Server
uvicorn backend.app.main:app --reload --port 8000
```
API docs available at: `http://localhost:8000/docs`  
Health check: `http://localhost:8000/api/health`

### 2. Frontend Setup
```bash
# From project root
cd frontend
npm install
npm run dev
```
Dashboard available at: `http://localhost:5173`

---

## 🧪 Pipeline Test Script
Run full pipeline validation script:
```bash
python scripts/test_pipeline.py
```

---

## Phase Roadmap
- [x] **Phase 0**: Project Scaffolding & Setup
- [x] **Phase 1**: Signal Preprocessing & FFT Engine
- [x] **Phase 2**: VMD Decomposition Engine
- [x] **Phase 3**: Adaptive VMD Parameter Search
- [x] **Phase 4**: Feature Extraction & Schema Layer
- [x] **Phase 5**: ML Model Training & Serialization
- [x] **Phase 6**: Explainability Layer (SHAP)
- [x] **Phase 7**: End-to-End API Integration
- [x] **Phase 8**: Industrial Frontend Dashboard & Pages
- [x] **Phase 9**: Polish, Verification & Documentation

## What Is Real Today
The backend performs signal validation, preprocessing, FFT/statistics, fixed and adaptive
VMD, feature extraction, model inference, SHAP explanation, and SQLite history persistence.
The frontend is an API-backed control-room interface with explicit loading, empty, error,
and pending-integration states. Sample bearing CSVs are synthetic project fixtures and are
labelled as demo data; they are not a substitute for certified industrial monitoring.

To train a new model from the included sample fixtures:

```bash
python scripts/train_model.py
```

The active model bundle is stored in `models/`, while prior bundles are archived in
`models/versions/`. The health score is an engineering heuristic, and SHAP values describe
model contribution rather than physical causation.

## Operator Runbook
For Windows setup commands, service startup, verification, CSV analysis, and instructions
for connecting VMD Explorer and Model Training, see:

- [PDF runbook](docs/VibraGuard_Runbook.pdf)
- [Editable runbook source](docs/runbook.md)
