import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from ml.features.feature_extractor import FeatureExtractor


class VibraGuardPredictor:
    """
    VibraGuard ML inference pipeline.

    Pipeline:
        vibration signal
            -> FeatureExtractor
            -> ordered 101-feature vector
            -> scaler
            -> Random Forest
            -> label encoder
    """

    def __init__(
        self,
        model_dir: str = "models/versions/v1.0.3",
        K: int = 5,
        alpha: float = 2000.0,
    ):
        self.model_dir = Path(model_dir)

        self.K = K
        self.alpha = alpha

        # Load trained artifacts
        self.model = joblib.load(self.model_dir / "model.joblib")
        self.scaler = joblib.load(self.model_dir / "scaler.joblib")
        self.label_encoder = joblib.load(
            self.model_dir / "label_encoder.joblib"
        )

        # Load feature schema
        with open(
            self.model_dir / "feature_schema.json",
            "r",
            encoding="utf-8",
        ) as f:
            schema = json.load(f)

        self.feature_names = schema["feature_names"]

        # Feature extractor
        self.extractor = FeatureExtractor(
            default_K=self.K,
            default_alpha=self.alpha,
        )

        # Safety check
        if len(self.feature_names) != self.model.n_features_in_:
            raise ValueError(
                f"Feature mismatch: schema contains "
                f"{len(self.feature_names)} features, but model expects "
                f"{self.model.n_features_in__}."
            )

    def predict_signal(
        self,
        signal: np.ndarray,
        fs: float = 12000.0,
    ):
        """
        Predict machine condition from a vibration signal.
        """

        # Extract features
        feature_dict = self.extractor.extract_feature_dict(
            signal=signal,
            fs=fs,
        )

        # Make sure all expected features exist
        missing = [
            name
            for name in self.feature_names
            if name not in feature_dict
        ]

        if missing:
            raise ValueError(
                f"Missing {len(missing)} features. "
                f"First missing features: {missing[:10]}"
            )

        # IMPORTANT:
        # Arrange features in exactly the same order used during training.
        feature_vector = np.array(
            [feature_dict[name] for name in self.feature_names],
            dtype=np.float64,
        ).reshape(1, -1)

        # Scale
        scaled_features = self.scaler.transform(feature_vector)

        # Prediction
        prediction_encoded = self.model.predict(scaled_features)[0]

        prediction_label = self.label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        # Probabilities
        probabilities = self.model.predict_proba(
            scaled_features
        )[0]

        class_probabilities = {}

        for class_name, probability in zip(
            self.label_encoder.classes_,
            probabilities,
        ):
            class_probabilities[str(class_name)] = float(probability)

        confidence = float(np.max(probabilities))

        return {
            "prediction": str(prediction_label),
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "feature_count": len(feature_vector[0]),
            "sampling_frequency_hz": fs,
            "vmd_parameters": {
                "K": self.K,
                "alpha": self.alpha,
            },
        }

    def predict_csv(
        self,
        csv_path: str,
        fs: float = 12000.0,
    ):
        """
        Load a vibration CSV and run prediction.
        """

        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {path}"
            )

        df = pd.read_csv(path)

        # Find numeric columns
        numeric_columns = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if not numeric_columns:
            raise ValueError(
                "CSV does not contain a numeric vibration signal."
            )

        # Use first numeric column
        signal = df[numeric_columns[0]].to_numpy(
            dtype=np.float64
        )

        result = self.predict_signal(
            signal=signal,
            fs=fs,
        )

        result["signal"] = {
            "file": str(path),
            "samples": len(signal),
            "column_used": numeric_columns[0],
        }

        return result