# VibraGuard Runbook

## Purpose

This runbook explains how to install, start, verify, and extend the VibraGuard predictive-maintenance prototype on Windows PowerShell.

The included sample CSV files are synthetic project fixtures. They are clearly demo data and must not be presented as certified industrial measurements.

## 1. Install

Open PowerShell in `D:\mfc-anti`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Set-Location frontend
npm install
Set-Location ..
```

If PowerShell blocks activation for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2. Start the services

Use two PowerShell windows.

Backend window, from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Frontend window, from the repository root:

```powershell
Set-Location frontend
npm run dev -- --host 127.0.0.1
```

Open:

- Dashboard: http://127.0.0.1:5173/
- API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/api/health

## 3. Verify the installation

From the repository root:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
python scripts/test_pipeline.py
Set-Location frontend
npm run build
Set-Location ..
```

The full pipeline test verifies scaffolding, preprocessing, FFT/statistics, fixed VMD, adaptive VMD, features, training, SHAP, prediction, history, and system status.

## 4. Train or refresh a model

The included training script uses the sample healthy and faulty bearing fixtures:

```powershell
python scripts/train_model.py
```

Supported model names in the Python trainer are:

- `Random Forest`
- `Gradient Boosting`
- `SVM`
- `Logistic Regression`

The active bundle is written to `models\`:

- `model.joblib`
- `scaler.joblib`
- `feature_schema.json`
- `training_metadata.json`
- `label_encoder.joblib`

Previous bundles are archived under `models\versions\vX.Y.Z\`.

## 5. Analyze a CSV signal

The CSV needs a numeric `vibration` column, or another numeric column that can be selected explicitly by the adapter. Minimum length is 128 samples.

The browser flow is:

1. Open Signal Lab.
2. Select a `.csv` or `.txt` file.
3. Inspect processed waveform, FFT preview, RMS, and dominant frequency.
4. Select Run diagnosis to call the prediction API with the same file.

PowerShell API example:

```powershell
$form = @{ file = Get-Item .\ml\data\sample\faulty_bearing.csv }
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/signal/upload -Method Post -Form $form
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/predict/file -Method Post -Form $form
```

Unknown RPM, load, and machine ID values remain `Unknown / Not Provided`; they are never invented.

## 6. Connect VMD Explorer

The backend endpoints are already available:

```text
POST /api/v1/vmd/decompose
POST /api/v1/vmd/optimize
```

Both accept JSON with at least 128 signal samples:

```json
{
  "signal": [0.1, 0.2, 0.0],
  "sampling_rate": 12000,
  "K": 5,
  "alpha": 2000,
  "tau": 0,
  "DC": 0
}
```

A real frontend VMD Explorer should:

1. Reuse the selected signal kept by Signal Lab.
2. Call `/api/v1/vmd/optimize` for adaptive mode.
3. Render `initial_K`, `initial_alpha`, `initial_fitness`, `optimized_K`, `optimized_alpha`, and `optimized_fitness`.
4. Plot each `optimized_decomposition.imf_waveforms` series.
5. Render `imf_stats` for RMS, energy, kurtosis, dominant frequency, entropy, and center frequency.
6. Plot `trajectory` as fitness against candidate K/alpha values.
7. Label the result as a measured decomposition; never invent convergence or fitness values.

Example TypeScript call:

```ts
const response = await fetch('/api/v1/vmd/optimize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ signal, sampling_rate: 12000, k_min: 3, k_max: 7 }),
});
const result = await response.json();
```

The current UI labels VMD Explorer as `SERVER INTEGRATION PENDING`; it does not fabricate IMF results.

## 7. Connect Model Training

The backend training endpoint is:

```text
POST /api/v1/system/train
```

Request body:

```json
{ "model_name": "Random Forest" }
```

PowerShell example:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/system/train -Method Post -ContentType 'application/json' -Body '{"model_name":"Random Forest"}'
```

A real frontend training page should:

1. Provide a model selector using the four supported names.
2. POST the selected model to `/api/v1/system/train`.
3. Show a loading state while training runs.
4. Render returned metadata: version, accuracy, precision, recall, macro-F1, ROC-AUC, confusion matrix, and sample counts.
5. Refresh `/api/v1/system/status` after success.
6. Show failures as actionable errors, not raw stack traces.

The current UI labels Model Training as `SERVER INTEGRATION PENDING`; no training metrics are fabricated.

## 8. Useful API calls

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/system/status
Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/predict/history?limit=20'
```

Interactive Swagger testing is available at http://127.0.0.1:8000/docs.

## 9. Troubleshooting

**Frontend says API OFFLINE**

- Confirm Uvicorn is running on port 8000.
- Confirm Vite is running on port 5173.
- Check that the frontend proxy in `frontend\vite.config.ts` targets `http://localhost:8000`.

**Upload returns 400 or 422**

- Use `.csv` or `.txt`.
- Include at least 128 numeric vibration samples.
- Remove NaN and infinite values.
- Provide a numeric vibration column.

**Model files are missing**

```powershell
python scripts/train_model.py
```

**The model score is not a calibrated probability**

Treat confidence as a model score unless the model and calibration procedure explicitly guarantee calibrated probabilities.

## 10. Engineering disclaimers

The machine health score is an adjustable engineering heuristic, not a certified safety metric. SHAP values describe which features contributed most strongly to a model prediction; they do not prove physical causation. Adaptive VMD optimization is measured by the documented fitness function and must not be described as improving classification accuracy without a controlled experiment.
