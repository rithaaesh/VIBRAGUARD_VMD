import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional


class SHAPExplainer:
    """
    Computes local and global SHAP (SHapley Additive exPlanations) values
    and maps feature importances to source Intrinsic Mode Functions (IMFs).
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = joblib.load(os.path.join(model_dir, "model.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        self.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

        with open(os.path.join(model_dir, "feature_schema.json"), "r") as f:
            self.feature_schema = json.load(f)

        self.feature_names = self.feature_schema["feature_names"]

        # Initialize SHAP TreeExplainer or KernelExplainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            # Fallback to KernelExplainer if model is not tree-based
            dummy_bg = np.zeros((5, len(self.feature_names)))
            self.explainer = shap.KernelExplainer(self.model.predict_proba, dummy_bg)

    def explain_sample(
        self, feature_dict: Dict[str, float], top_k: int = 8
    ) -> Dict[str, Any]:
        """
        Computes SHAP attribution values for a single prediction instance.
        Maps feature contributions back to individual IMFs and raw signal components.
        """
        # Align feature values with schema ordering
        feat_vector = pd.DataFrame(
            [[float(feature_dict.get(fname, 0.0)) for fname in self.feature_names]],
            columns=self.feature_names,
        )

        # Scale features using saved fitted scaler
        feat_scaled = self.scaler.transform(feat_vector)

        # Compute SHAP values
        shap_values = self.explainer.shap_values(feat_scaled)

        # Handle multi-class / binary array shapes
        if isinstance(shap_values, list):
            # Pick predicted class shap array
            pred_idx = self.model.predict(feat_scaled)[0]
            vals = shap_values[pred_idx][0]
        elif shap_values.ndim == 3:
            pred_idx = self.model.predict(feat_scaled)[0]
            vals = shap_values[0, :, pred_idx]
        else:
            vals = shap_values[0]

        # Top local feature contributions
        abs_vals = np.abs(vals)
        sorted_indices = np.argsort(abs_vals)[::-1]

        top_features = []
        for idx in sorted_indices[:top_k]:
            fname = self.feature_names[idx]
            val = float(vals[idx])
            raw_val = float(feature_dict.get(fname, 0.0))

            top_features.append(
                {
                    "feature_name": fname,
                    "shap_value": val,
                    "importance": abs(val),
                    "feature_value": raw_val,
                    "impact_direction": "Positive" if val >= 0 else "Negative",
                    "explanation_text": f"{fname} ({raw_val:.4f}) contributed {'positively' if val >= 0 else 'negatively'} to predicted class.",
                }
            )

        # Map to IMF level attribution
        imf_attribution: Dict[str, float] = {}
        for fname, val in zip(self.feature_names, vals):
            imf_prefix = fname.split("_")[0] + "_" + fname.split("_")[1] if fname.startswith("imf_") else "raw_signal"
            imf_attribution[imf_prefix] = imf_attribution.get(imf_prefix, 0.0) + abs(float(val))

        # Normalize IMF attribution percentages
        total_attrib = sum(imf_attribution.values())
        if total_attrib > 1e-12:
            imf_attribution_pct = {
                k: float((v / total_attrib) * 100.0) for k, v in imf_attribution.items()
            }
        else:
            imf_attribution_pct = {k: 0.0 for k in imf_attribution}

        return {
            "top_features": top_features,
            "imf_attribution": imf_attribution_pct,
            "disclaimer": "These features contributed most strongly to the model's prediction according to SHAP feature attribution; this does not prove physical causation.",
        }

    def explain_global(
        self, feature_dicts: List[Dict[str, float]], top_k: int = 10
    ) -> Dict[str, Any]:
        """Aggregates real SHAP values across a supplied dataset of feature rows."""
        if not feature_dicts:
            raise ValueError("At least one feature row is required for global explanation.")

        feat_matrix = pd.DataFrame(
            [
                [float(feature_dict.get(fname, 0.0)) for fname in self.feature_names]
                for feature_dict in feature_dicts
            ],
            columns=self.feature_names,
        )
        feat_scaled = self.scaler.transform(feat_matrix)
        shap_values = self.explainer.shap_values(feat_scaled)

        if isinstance(shap_values, list):
            values = np.stack([np.asarray(class_values) for class_values in shap_values], axis=0)
            mean_importance = np.mean(np.abs(values), axis=(0, 2))
        elif np.asarray(shap_values).ndim == 3:
            values = np.asarray(shap_values)
            mean_importance = np.mean(np.abs(values), axis=(0, 2))
        else:
            mean_importance = np.mean(np.abs(np.asarray(shap_values)), axis=0)

        sorted_indices = np.argsort(mean_importance)[::-1][:top_k]
        top_features = [
            {
                "feature_name": self.feature_names[idx],
                "mean_abs_shap": float(mean_importance[idx]),
            }
            for idx in sorted_indices
        ]
        return {
            "top_features": top_features,
            "sample_count": len(feature_dicts),
            "disclaimer": "These features contributed most strongly to model predictions across the supplied dataset according to SHAP attribution; this does not prove physical causation.",
        }
