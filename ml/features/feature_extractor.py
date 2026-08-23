import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from ml.features.signal_stats import SignalStatsCalculator
from ml.preprocessing.signal_preprocessor import SignalPreprocessor
from ml.vmd.vmd_engine import VMDEngine


class FeatureExtractor:
    """
    Unified feature extractor combining raw signal time/frequency features
    and decomposed IMF-level condition indicators into a standard feature vector.
    """

    def __init__(self, default_K: int = 5, default_alpha: float = 2000.0):
        self.default_K = default_K
        self.default_alpha = default_alpha

    def extract_feature_dict(
        self,
        signal: np.ndarray,
        fs: float = 12000.0,
        imf_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Extracts named feature dictionary for a single 1D signal window.
        """
        signal = SignalPreprocessor.validate_signal(signal)

        features: Dict[str, float] = {}

        # 1. Raw Signal Time-Domain Features (11 features)
        raw_time_feats = SignalStatsCalculator.compute_time_features(signal)
        for name, val in raw_time_feats.items():
            features[f"raw_{name}"] = float(val)

        # 2. Raw Signal Frequency-Domain Features (5 features)
        raw_freq_feats = SignalStatsCalculator.compute_frequency_features(signal, fs)
        for name, val in raw_freq_feats.items():
            features[f"raw_{name}"] = float(val)

        # 3. IMF-Level Features (K * 16 features)
        if imf_matrix is None:
            engine = VMDEngine(K=self.default_K, alpha=self.default_alpha)
            decomp = engine.decompose(signal, fs=fs)
            imf_matrix = decomp["raw_imfs"]
        else:
            imf_matrix = np.asarray(imf_matrix, dtype=np.float64)
            if imf_matrix.ndim != 2 or imf_matrix.shape[1] != len(signal):
                raise ValueError("IMF matrix must have shape (K, signal_length).")
            if not np.isfinite(imf_matrix).all():
                raise ValueError("IMF matrix contains NaN or infinite values; recompute the decomposition.")

        num_imfs = imf_matrix.shape[0]
        for k in range(num_imfs):
            imf_k = imf_matrix[k, :]
            prefix = f"imf_{k+1}"

            # IMF Time Features
            imf_time = SignalStatsCalculator.compute_time_features(imf_k)
            for name, val in imf_time.items():
                features[f"{prefix}_{name}"] = float(val)

            # IMF Freq Features
            imf_freq = SignalStatsCalculator.compute_frequency_features(imf_k, fs)
            for name, val in imf_freq.items():
                features[f"{prefix}_{name}"] = float(val)

            # Energy ratio
            tot_energy = sum(np.sum(imf_matrix[i] ** 2) for i in range(num_imfs))
            imf_energy = np.sum(imf_k**2)
            features[f"{prefix}_energy_ratio"] = float(imf_energy / tot_energy) if tot_energy > 1e-12 else 0.0

        if not all(np.isfinite(value) for value in features.values()):
            raise ValueError("Feature extraction produced a non-finite value; check the input signal.")

        return features

    def get_feature_names(self, K: Optional[int] = None) -> List[str]:
        """Returns ordered list of feature names for K IMFs."""
        target_K = K or self.default_K
        dummy_signal = np.sin(np.linspace(0, 10, 1024))
        dummy_imfs = VMDEngine(K=target_K, alpha=self.default_alpha).decompose(dummy_signal)["raw_imfs"]
        feats = self.extract_feature_dict(dummy_signal, imf_matrix=dummy_imfs)
        return list(feats.keys())

    def save_feature_schema(
        self, output_path: str = "models/feature_schema.json", K: Optional[int] = None
    ):
        """Saves JSON feature schema to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        feature_names = self.get_feature_names(K)
        schema = {
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "version": "1.0.0",
        }
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"Saved feature schema ({len(feature_names)} features) -> {output_path}")
        return schema
