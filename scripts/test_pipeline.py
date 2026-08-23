"""
VibraGuard Pipeline Test Harness
Phases 0, 1, 2, 3, 4, 5, 6 & 7 End-to-End Verification Script
"""

import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VibraGuardTest")


def run_phase_0_checks():
    logger.info("--- Running Phase 0: System Scaffolding Verification ---")
    try:
        from backend.app.main import app
        from backend.app.core.config import settings
        logger.info(f"✓ Backend app initialized cleanly (System: {settings.APP_NAME})")
        return True
    except Exception as e:
        logger.error(f"✗ Failed Phase 0 checks: {e}")
        return False


def run_phase_1_checks():
    logger.info("--- Running Phase 1: Signal Engine Verification ---")
    try:
        from ml.preprocessing.dataset_adapter import GenericCSVAdapter
        from ml.preprocessing.signal_preprocessor import SignalPreprocessor
        from fastapi.testclient import TestClient
        from backend.app.main import app

        adapter = GenericCSVAdapter()
        raw_signal, _ = adapter.load_signal("ml/data/sample/healthy_bearing.csv")
        preprocessor = SignalPreprocessor()
        prep_res = preprocessor.process(raw_signal)

        client = TestClient(app)
        res = client.post("/api/v1/signal/process", json={"signal": prep_res["processed_signal"][:1024].tolist()})
        assert res.status_code == 200
        logger.info("✓ Phase 1 Signal Engine checks PASSED!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed Phase 1 checks: {e}", exc_info=True)
        return False


def run_phase_2_checks():
    logger.info("--- Running Phase 2: Fixed VMD Engine Verification ---")
    try:
        from ml.vmd.vmd_engine import VMDEngine
        from ml.preprocessing.dataset_adapter import GenericCSVAdapter

        adapter = GenericCSVAdapter()
        raw_signal, _ = adapter.load_signal("ml/data/sample/healthy_bearing.csv")
        vmd_engine = VMDEngine(K=4, alpha=2000.0)
        res = vmd_engine.decompose(raw_signal[:2048])
        assert res["K"] == 4
        assert res["alpha"] == 2000.0
        assert len(res["imf_stats"]) == 4
        assert len(res["imf_waveforms"]) == 4
        assert res["raw_imfs"].shape == (4, 2048)
        assert res["reconstruction_error"] < 0.15, (
            f"Reconstruction error {res['reconstruction_error']:.4f} exceeds 0.15"
        )
        logger.info("✓ Phase 2 VMD Engine checks PASSED!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed Phase 2 checks: {e}", exc_info=True)
        return False


def run_phase_3_checks():
    logger.info("--- Running Phase 3: Adaptive VMD Optimization Verification ---")
    try:
        from ml.optimization.vmd_optimizer import VMDOptimizer
        from ml.preprocessing.dataset_adapter import GenericCSVAdapter

        adapter = GenericCSVAdapter()
        raw_signal, _ = adapter.load_signal("ml/data/sample/faulty_bearing.csv")
        optimizer = VMDOptimizer()
        opt_res = optimizer.optimize_deterministic(raw_signal[:2048], k_range=(3, 5))
        assert opt_res["optimized_fitness"] <= opt_res["initial_fitness"]
        assert opt_res["trajectory"]
        assert opt_res["trajectory"][0]["K"] == opt_res["initial_K"]
        assert opt_res["trajectory"][0]["alpha"] == opt_res["initial_alpha"]
        assert all(np.isfinite(point["fitness"]) for point in opt_res["trajectory"])
        logger.info("✓ Phase 3 Adaptive VMD checks PASSED!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed Phase 3 checks: {e}", exc_info=True)
        return False


def run_phase_4_checks():
    logger.info("--- Running Phase 4: Feature Extraction Verification ---")
    try:
        from ml.features.feature_extractor import FeatureExtractor
        from ml.preprocessing.dataset_adapter import GenericCSVAdapter

        adapter = GenericCSVAdapter()
        raw_signal, _ = adapter.load_signal("ml/data/sample/healthy_bearing.csv")

        extractor = FeatureExtractor(default_K=4)
        feat_dict = extractor.extract_feature_dict(raw_signal[:1024])
        assert len(feat_dict) > 50
        schema = extractor.save_feature_schema("models/feature_schema.json", K=4)
        assert os.path.exists("models/feature_schema.json")
        logger.info("✓ Phase 4 Feature Extraction checks PASSED!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed Phase 4 checks: {e}", exc_info=True)
        return False


def run_phase_5_checks():
    logger.info("--- Running Phase 5: ML Training & Model Serialization Verification ---")
    try:
        from ml.training.trainer import ModelTrainer

        trainer = ModelTrainer(model_dir="models")
        meta = trainer.train_and_evaluate("Random Forest")
        required_artifacts = (
            "model.joblib",
            "scaler.joblib",
            "feature_schema.json",
            "training_metadata.json",
            "label_encoder.joblib",
        )
        assert all(os.path.exists(os.path.join("models", artifact)) for artifact in required_artifacts)
        assert len(meta["classes"]) == 2
        assert meta["macro_f1"] >= 0.0
        assert meta["roc_auc"] is not None
        assert meta["version"].startswith("v")
        logger.info("✓ Phase 5 ML Training checks PASSED!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed Phase 5 checks: {e}", exc_info=True)
        return False


def run_phase_6_checks():
    logger.info("--- Running Phase 6: SHAP Explainability Verification ---")
    try:
        from ml.explainability.shap_explainer import SHAPExplainer
        from ml.features.feature_extractor import FeatureExtractor
        from ml.preprocessing.dataset_adapter import GenericCSVAdapter

        adapter = GenericCSVAdapter()
        raw_signal, _ = adapter.load_signal("ml/data/sample/faulty_bearing.csv")
        extractor = FeatureExtractor()
        feat_dict = extractor.extract_feature_dict(raw_signal[:1024])

        explainer = SHAPExplainer(model_dir="models")
        shap_res = explainer.explain_sample(feat_dict, top_k=5)
        assert len(shap_res["top_features"]) == 5
        healthy_signal, _ = adapter.load_signal("ml/data/sample/healthy_bearing.csv")
        global_features = [
            extractor.extract_feature_dict(healthy_signal[:1024]),
            feat_dict,
        ]
        global_res = explainer.explain_global(global_features, top_k=5)
        assert len(global_res["top_features"]) == 5
        assert global_res["sample_count"] == 2
        assert all(point["mean_abs_shap"] >= 0.0 for point in global_res["top_features"])
        logger.info("✓ Phase 6 SHAP Explainability checks PASSED!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed Phase 6 checks: {e}", exc_info=True)
        return False


def run_phase_7_checks():
    logger.info("--- Running Phase 7: End-to-End API & Database Integration Verification ---")
    try:
        from fastapi.testclient import TestClient
        from backend.app.main import app
        from ml.preprocessing.dataset_adapter import GenericCSVAdapter

        adapter = GenericCSVAdapter()
        raw_signal, _ = adapter.load_signal("ml/data/sample/faulty_bearing.csv")

        client = TestClient(app)

        # 1. E2E Predict Endpoint Test
        res = client.post(
            "/api/v1/predict",
            json={
                "signal": raw_signal[:1024].tolist(),
                "machine_id": "PUMP-TEST-7",
                "filename": "faulty_bearing.csv",
                "adaptive_vmd": True,
            },
        )
        assert res.status_code == 200, f"Predict API returned {res.status_code}: {res.text}"
        data = res.json()

        assert "predicted_fault" in data and "confidence" in data and "health_score" in data, "Missing prediction keys"
        assert "top_features" in data and "imf_attribution" in data, "Missing XAI keys"
        assert data["prediction_id"] is not None, "Prediction ID not returned from database"

        logger.info(f"✓ E2E Predict Endpoint: Fault='{data['predicted_fault']}' (Conf: {data['confidence']*100:.1f}%, Health: {data['health_score']:.1f}, ID: {data['prediction_id']})")

        # 1b. CSV file inference and invalid-format handling
        with open("ml/data/sample/faulty_bearing.csv", "rb") as signal_file:
            file_res = client.post(
                "/api/v1/predict/file",
                files={"file": ("faulty_bearing.csv", signal_file, "text/csv")},
            )
        assert file_res.status_code == 200, f"File predict API returned {file_res.status_code}: {file_res.text}"
        invalid_file_res = client.post(
            "/api/v1/predict/file",
            files={"file": ("signal.json", b"{}", "application/json")},
        )
        assert invalid_file_res.status_code == 400

        # 2. History Endpoint Test
        hist_res = client.get("/api/v1/predict/history")
        assert hist_res.status_code == 200, f"History API returned {hist_res.status_code}"
        history = hist_res.json()
        assert len(history) > 0, "No records returned from history"
        logger.info(f"✓ History Endpoint returned {len(history)} persistent DB prediction records")

        # 3. System Status Endpoint Test
        sys_res = client.get("/api/v1/system/status")
        assert sys_res.status_code == 200
        sys_data = sys_res.json()
        assert sys_data["status"] == "online" and sys_data["active_model"] == "Random Forest"
        assert sys_data["feature_count"] > 0
        logger.info("✓ System Status Endpoint verified online & active model metadata")

        logger.info("✓ Phase 7 API & Database Integration checks PASSED!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed Phase 7 checks: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    if not run_phase_0_checks(): sys.exit(1)
    if not run_phase_1_checks(): sys.exit(1)
    if not run_phase_2_checks(): sys.exit(1)
    if not run_phase_3_checks(): sys.exit(1)
    if not run_phase_4_checks(): sys.exit(1)
    if not run_phase_5_checks(): sys.exit(1)
    if not run_phase_6_checks(): sys.exit(1)
    if not run_phase_7_checks(): sys.exit(1)

    print("\n[SUCCESS] ALL PHASE 0 THROUGH PHASE 7 PIPELINE TESTS PASSED SUCCESSFULLY!")
