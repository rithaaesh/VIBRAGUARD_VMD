import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Any


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MCP IMPORTS
# ============================================================

from mcp.server.mcpserver import MCPServer

from ml.vmd.vmd_engine import VMDEngine
from ml.inference.predictor import VibraGuardPredictor


# ============================================================
# MCP SERVER
# ============================================================

mcp = MCPServer("VibraGuard")

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def _active_model_dir() -> Path:
    """Prefer the active bundle and fall back to the newest archived bundle."""
    active = PROJECT_ROOT / "models"
    if (active / "model.joblib").exists():
        return active
    versions = sorted((active / "versions").glob("v*"))
    if not versions:
        raise FileNotFoundError("No trained model bundle found in models/.")
    return versions[-1]


try:
    model_dir = _active_model_dir()
    predictor = VibraGuardPredictor(model_dir=str(model_dir), K=5, alpha=2000.0)
    MODEL_LOADED = True
    MODEL_ERROR = None
    MODEL_VERSION = model_dir.name if model_dir.parent.name == "versions" else "active"
except Exception as e:

    predictor = None
    MODEL_LOADED = False
    MODEL_ERROR = str(e)
    MODEL_VERSION = "unavailable"


def _load_signal(csv_path: str) -> tuple[Path, np.ndarray, str]:
    path = Path(csv_path).resolve()
    if not path.exists() or path.suffix.lower() not in {".csv", ".txt"}:
        raise ValueError("A readable .csv or .txt signal file is required.")
    df = pd.read_csv(path)
    preferred = [column for column in df.columns if column.lower() in {"vibration", "signal", "value", "acceleration", "amplitude"}]
    numeric = preferred or df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        raise ValueError("No numeric vibration column found in CSV.")
    column = numeric[0]
    signal = df[column].to_numpy(dtype=np.float64)
    if len(signal) < 128:
        raise ValueError(f"Signal contains only {len(signal)} samples; at least 128 are required.")
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or infinite values.")
    return path, signal, column


# ============================================================
# TOOL 1 — SERVER HEALTH CHECK
# ============================================================

@mcp.tool()
def extract_features(csv_path: str, fs: float = 12000.0) -> Dict[str, Any]:
    """Extract the real time, frequency, and IMF feature vector from a CSV signal."""
    try:
        path, signal, column = _load_signal(csv_path)
        features = predictor.extractor.extract_feature_dict(signal, fs=fs) if predictor else None
        if features is None:
            return {"status": "error", "message": "No trained feature pipeline is available."}
        return {"status": "success", "signal": {"file": str(path), "column": column, "samples": len(signal)}, "feature_count": len(features), "features": features}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def diagnose_fault(csv_path: str, fs: float = 12000.0) -> Dict[str, Any]:
    """Run the trained classifier and return a real fault diagnosis with confidence."""
    return diagnose_machine(csv_path, fs)


@mcp.tool()
def calculate_health(prediction: str, confidence: float, rms: float, kurtosis: float) -> Dict[str, Any]:
    """Calculate the documented engineering health heuristic; this is not a certified metric."""
    confidence = max(0.0, min(1.0, float(confidence)))
    health = 100.0
    if prediction.lower() != "normal":
        health -= 45.0 + confidence * 15.0
        severity = "HIGH SEVERITY"
    else:
        health -= (1.0 - confidence) * 10.0
        severity = "HEALTHY"
    if kurtosis > 4.0:
        health -= min(15.0, (kurtosis - 4.0) * 3.0)
    return {"status": "success", "health_score": max(0.0, min(100.0, health)), "severity": severity, "disclaimer": "Engineering heuristic, not a certified safety metric."}


@mcp.tool()
def estimate_rul(csv_path: str) -> Dict[str, Any]:
    """Report RUL availability without inventing a days-to-failure estimate."""
    return {"status": "unavailable", "estimate": None, "message": "RUL is not implemented: historical degradation data and a dedicated RUL model are required."}


@mcp.tool()
def analyze_machine(csv_path: str, fs: float = 12000.0) -> Dict[str, Any]:
    """Run the complete signal, VMD, feature, diagnosis, and health analysis."""
    diagnosis = diagnose_machine(csv_path, fs)
    vmd = run_vmd(csv_path, fs)
    if diagnosis.get("status") != "success":
        return diagnosis
    diagnosis["vmd"] = vmd
    diagnosis["feature_count"] = predictor.model.n_features_in_ if predictor else None
    diagnosis["rul"] = estimate_rul(csv_path)
    return diagnosis

@mcp.tool()
def hello_vibraguard() -> dict:
    """
    Check whether the VibraGuard MCP server is running.
    """

    return {
        "status": "success",
        "result": "VibraGuard MCP Server is working successfully!",
        "model_loaded": MODEL_LOADED,
        "model_version": MODEL_VERSION,
    }


# ============================================================
# TOOL 2 — RUN VMD
# ============================================================

@mcp.tool()
def run_vmd(
    csv_path: str,
    fs: float = 12000.0,
    K: int = 5,
    alpha: float = 2000.0,
) -> Dict[str, Any]:
    """
    Run Variational Mode Decomposition on a vibration CSV file.

    CSV
        ↓
    Vibration Signal
        ↓
    VMD
        ↓
    IMF decomposition
        ↓
    IMF statistics
    """

    try:

        # ----------------------------------------------------
        # 1. Validate file
        # ----------------------------------------------------

        path = Path(csv_path)

        if not path.exists():

            return {
                "status": "error",
                "message": f"CSV file not found: {csv_path}",
            }

        # ----------------------------------------------------
        # 2. Load CSV
        # ----------------------------------------------------

        df = pd.read_csv(path)

        numeric_columns = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if not numeric_columns:

            return {
                "status": "error",
                "message": (
                    "No numeric vibration column found in CSV."
                ),
            }

        # Use first numeric column

        signal = df[numeric_columns[0]].to_numpy(
            dtype=np.float64
        )

        # ----------------------------------------------------
        # 3. Validate signal
        # ----------------------------------------------------

        if len(signal) < 128:

            return {
                "status": "error",
                "message": (
                    f"Signal contains only {len(signal)} samples. "
                    "At least 128 samples are required."
                ),
            }

        if not np.isfinite(signal).all():

            return {
                "status": "error",
                "message": (
                    "Signal contains NaN or infinite values."
                ),
            }

        # ----------------------------------------------------
        # 4. Run VMD
        # ----------------------------------------------------

        engine = VMDEngine(
            K=K,
            alpha=alpha,
        )

        result = engine.decompose(
            signal=signal,
            fs=fs,
        )

        # ----------------------------------------------------
        # 5. Remove raw numpy matrix
        # MCP responses must be JSON serializable.
        # ----------------------------------------------------

        result.pop("raw_imfs", None)

        # ----------------------------------------------------
        # 6. Add signal information
        # ----------------------------------------------------

        result["status"] = "success"

        result["signal"] = {

            "file": str(path),

            "column_used": numeric_columns[0],

            "samples": int(len(signal)),

            "sampling_frequency_hz": float(fs),
        }

        # ----------------------------------------------------
        # 7. Add decomposition quality
        # ----------------------------------------------------

        result["decomposition"] = {

            "modes": int(K),

            "alpha": float(alpha),

            "reconstruction_error": float(
                result.get(
                    "reconstruction_error",
                    0.0,
                )
            ),
        }

        return result

    except Exception as e:

        return {

            "status": "error",

            "error_type": type(e).__name__,

            "message": str(e),
        }


# ============================================================
# TOOL 3 — MACHINE DIAGNOSIS
# ============================================================

@mcp.tool()
def diagnose_machine(
    csv_path: str,
    fs: float = 12000.0,
) -> Dict[str, Any]:
    """
    Complete VibraGuard machine diagnosis.

    Pipeline:

    CSV
      ↓
    Vibration Signal
      ↓
    VMD
      ↓
    Feature Extraction
      ↓
    101 Features
      ↓
    Random Forest
      ↓
    Fault Classification
      ↓
    Health Score
      ↓
    Maintenance Recommendation
    """

    try:

        # ====================================================
        # 1. CHECK MODEL
        # ====================================================

        if not MODEL_LOADED or predictor is None:

            return {

                "status": "error",

                "message": (
                    "Trained VibraGuard model could not "
                    "be loaded."
                ),

                "model_error": MODEL_ERROR,
            }

        # ====================================================
        # 2. CHECK CSV
        # ====================================================

        path = Path(csv_path)

        if not path.exists():

            return {

                "status": "error",

                "message": (
                    f"CSV file not found: {csv_path}"
                ),
            }

        # ====================================================
        # 3. LOAD SIGNAL
        # ====================================================

        df = pd.read_csv(path)

        preferred_columns = [

            "vibration",

            "signal",

            "value",

            "acceleration",

            "amplitude",

            "x",

            "sensor",
        ]

        signal_column = None

        # Try known vibration column names first

        for col in preferred_columns:

            if col in df.columns:

                if pd.api.types.is_numeric_dtype(
                    df[col]
                ):

                    signal_column = col

                    break

        # If not found, use first numeric column

        if signal_column is None:

            numeric_columns = df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            if not numeric_columns:

                return {

                    "status": "error",

                    "message": (
                        "No numeric vibration column "
                        "found in CSV."
                    ),
                }

            signal_column = numeric_columns[0]

        # Extract signal

        signal = (
            df[signal_column]
            .dropna()
            .to_numpy(dtype=np.float64)
        )

        # ====================================================
        # 4. SIGNAL VALIDATION
        # ====================================================

        if len(signal) < 128:

            return {

                "status": "error",

                "message": (
                    f"Signal too short: "
                    f"{len(signal)} samples."
                ),
            }

        if not np.isfinite(signal).all():

            return {

                "status": "error",

                "message": (
                    "Signal contains NaN or "
                    "infinite values."
                ),
            }

        # ====================================================
        # 5. BASIC VIBRATION METRICS
        # ====================================================

        rms = float(
            np.sqrt(
                np.mean(signal ** 2)
            )
        )

        peak = float(
            np.max(
                np.abs(signal)
            )
        )

        variance = float(
            np.var(signal)
        )

        # Kurtosis

        mean = float(
            np.mean(signal)
        )

        std = float(
            np.std(signal)
        )

        if std > 1e-12:

            kurtosis = float(
                np.mean(
                    (
                        (signal - mean)
                        / std
                    ) ** 4
                )
            )

        else:

            kurtosis = 0.0

        # ====================================================
        # 6. FREQUENCY ANALYSIS
        # ====================================================

        frequencies = np.fft.rfftfreq(
            len(signal),
            d=1.0 / fs,
        )

        spectrum = np.abs(
            np.fft.rfft(signal)
        )

        if len(spectrum) > 1:

            spectrum[0] = 0.0

            dominant_index = int(
                np.argmax(spectrum)
            )

            dominant_frequency = float(
                frequencies[
                    dominant_index
                ]
            )

        else:

            dominant_frequency = 0.0

        # ====================================================
        # 7. MACHINE LEARNING PREDICTION
        # ====================================================

        prediction_result = predictor.predict_csv(

            csv_path=str(path),

            fs=fs,
        )

        # ====================================================
        # 8. EXTRACT MODEL RESULT
        # ====================================================

        prediction = str(
            prediction_result.get(
                "prediction",
                "Unknown",
            )
        )

        confidence = float(
            prediction_result.get(
                "confidence",
                0.0,
            )
        )

        confidence_percent = round(
            confidence * 100,
            2,
        )

        # ====================================================
        # 9. PROBABILITIES
        # ====================================================

        probabilities = (
            prediction_result.get(
                "probabilities",
                {}
            )
        )

        # ====================================================
        # 10. HEALTH ASSESSMENT
        # ====================================================

        if prediction == "Normal":

            if confidence >= 0.90:

                health_status = "HEALTHY"

                health_score = round(
                    confidence * 100,
                    2,
                )

                severity = "LOW"

                recommendation = (
                    "Machine condition appears normal. "
                    "Continue normal operation and "
                    "routine vibration monitoring."
                )

            elif confidence >= 0.70:

                health_status = "HEALTHY - MONITOR"

                health_score = round(
                    confidence * 100,
                    2,
                )

                severity = "LOW"

                recommendation = (
                    "Machine is currently classified "
                    "as normal, but continued vibration "
                    "monitoring is recommended."
                )

            else:

                health_status = "UNCERTAIN"

                health_score = 50.0

                severity = "MEDIUM"

                recommendation = (
                    "Prediction confidence is low. "
                    "Collect additional vibration data."
                )

        elif prediction == "Inner Race Fault":

            if confidence >= 0.90:

                health_status = "CRITICAL"

                health_score = round(
                    (1.0 - confidence) * 100,
                    2,
                )

                severity = "CRITICAL"

                recommendation = (
                    "High-confidence inner race bearing "
                    "fault detected. Inspect the bearing "
                    "immediately."
                )

            elif confidence >= 0.70:

                health_status = "WARNING"

                health_score = round(
                    (1.0 - confidence) * 100,
                    2,
                )

                severity = "HIGH"

                recommendation = (
                    "Possible inner race bearing fault "
                    "detected. Schedule inspection and "
                    "continue monitoring."
                )

            else:

                health_status = "UNCERTAIN"

                health_score = 50.0

                severity = "MEDIUM"

                recommendation = (
                    "Fault prediction has low confidence. "
                    "Collect additional vibration data."
                )

        else:

            health_status = "UNKNOWN"

            health_score = 50.0

            severity = "MEDIUM"

            recommendation = (
                "Unknown machine condition. "
                "Further analysis is required."
            )

        # ====================================================
        # 11. MACHINE HEALTH
        # ====================================================

        machine_health = {

            "score_percent": health_score,

            "status": health_status,

            "severity": severity,

            "fault_detected": (
                prediction != "Normal"
            ),

            "maintenance_required": (
                prediction == "Inner Race Fault"
                and confidence >= 0.70
            ),
        }

        # ====================================================
        # 12. FINAL RESULT
        # ====================================================

        return {

            "status": "success",

            "machine_health": machine_health,

            "diagnosis": {

                "prediction": prediction,

                "confidence_percent": (
                    confidence_percent
                ),

                "probabilities": probabilities,

                "recommendation": recommendation,
            },

            "vibration_metrics": {

                "rms": round(
                    rms,
                    6,
                ),

                "peak": round(
                    peak,
                    6,
                ),

                "variance": round(
                    variance,
                    6,
                ),

                "kurtosis": round(
                    kurtosis,
                    6,
                ),

                "dominant_frequency_hz": round(
                    dominant_frequency,
                    2,
                ),
            },

            "signal": {

                "file": str(path),

                "column": signal_column,

                "samples": int(
                    len(signal)
                ),

                "sampling_frequency_hz": float(
                    fs
                ),
            },

            "model": {

                "type": "Random Forest",

                "version": MODEL_VERSION,

                "features": 101,
            },

            "maintenance": {

                "status": (
                    "IMMEDIATE ACTION"
                    if severity == "CRITICAL"
                    else
                    "INSPECTION RECOMMENDED"
                    if severity == "HIGH"
                    else
                    "CONTINUE MONITORING"
                ),

                "recommendation": recommendation,
            },

            "note": (
                "Health score is a diagnostic condition "
                "indicator. It is NOT a validated "
                "Remaining Useful Life (RUL) prediction. "
                "A real days-to-failure estimate requires "
                "historical degradation data and a dedicated "
                "RUL model."
            ),
        }

    except Exception as e:

        return {

            "status": "error",

            "error_type": type(e).__name__,

            "message": str(e),
        }


# ============================================================
# SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":

    mcp.run()