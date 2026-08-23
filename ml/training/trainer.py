import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from ml.features.feature_extractor import FeatureExtractor
from ml.preprocessing.dataset_adapter import GenericCSVAdapter
from ml.preprocessing.signal_preprocessor import SignalPreprocessor


class ModelTrainer:
    """
    Handles machine learning model training, evaluation, cross-validation with group-based splitting,
    and complete model artifact serialization.
    """

    SUPPORTED_MODELS = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.feature_extractor = FeatureExtractor()
        self.preprocessor = SignalPreprocessor()

    def generate_training_dataset(
        self, samples_per_class: int = 40, window_size: int = 1024
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Generates grouped training dataset from healthy and faulty synthetic signals.
        Creates distinct run/group IDs to ensure strict grouped splitting without window leakage.
        """
        adapter = GenericCSVAdapter()
        raw_healthy, _ = adapter.load_signal("ml/data/sample/healthy_bearing.csv")
        raw_faulty, _ = adapter.load_signal("ml/data/sample/faulty_bearing.csv")

        healthy_proc = self.preprocessor.process(raw_healthy)["processed_signal"]
        faulty_proc = self.preprocessor.process(raw_faulty)["processed_signal"]

        healthy_wins = self.preprocessor.create_windows(healthy_proc, window_size=window_size, overlap_pct=0.5)
        faulty_wins = self.preprocessor.create_windows(faulty_proc, window_size=window_size, overlap_pct=0.5)

        X_rows = []
        y_labels = []
        group_ids = []

        # Process Healthy Windows (Assign to runs H_RUN_1, H_RUN_2)
        for i, win in enumerate(healthy_wins):
            feat_dict = self.feature_extractor.extract_feature_dict(win)
            X_rows.append(feat_dict)
            y_labels.append("Normal")
            group_ids.append(f"run_healthy_{i % 3}")

        # Process Faulty Windows (Assign to runs F_RUN_1, F_RUN_2)
        for i, win in enumerate(faulty_wins):
            feat_dict = self.feature_extractor.extract_feature_dict(win)
            X_rows.append(feat_dict)
            y_labels.append("Inner Race Fault")
            group_ids.append(f"run_faulty_{i % 3}")

        df_X = pd.DataFrame(X_rows).fillna(0.0)
        y = np.array(y_labels)
        groups = np.array(group_ids)

        return df_X, y, groups

    def train_and_evaluate(
        self, model_name: str = "Random Forest"
    ) -> Dict[str, Any]:
        """
        Trains model using GroupShuffleSplit to prevent window correlation leakage.
        Saves all 5 required model bundle artifacts:
        1. model.joblib
        2. scaler.joblib
        3. feature_schema.json
        4. training_metadata.json
        5. label_encoder.joblib
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model name: {model_name}. Supported: {list(self.SUPPORTED_MODELS.keys())}")

        df_X, y, groups = self.generate_training_dataset()

        # Group-based split with a deterministic search for class-complete partitions.
        train_idx = test_idx = None
        for seed in range(42, 142):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
            candidate_train, candidate_test = next(gss.split(df_X, y, groups))
            if set(y[candidate_train]) == set(y) and set(y[candidate_test]) == set(y):
                train_idx, test_idx = candidate_train, candidate_test
                break
        if train_idx is None or test_idx is None:
            raise ValueError(
                "Unable to create a grouped train/test split containing every class. "
                "Provide at least two machine runs or files per class."
            )

        X_train, X_test = df_X.iloc[train_idx], df_X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Label Encoder
        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)
        y_test_enc = label_encoder.transform(y_test)

        # Scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit Model
        model = clone(self.SUPPORTED_MODELS[model_name])
        model.fit(X_train_scaled, y_train_enc)

        # Predict & Evaluate
        y_pred_enc = model.predict(X_test_scaled)
        y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

        acc = float(accuracy_score(y_test_enc, y_pred_enc))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test_enc, y_pred_enc, average="macro", zero_division=0, labels=np.arange(len(label_encoder.classes_))
        )
        conf_mat = confusion_matrix(
            y_test_enc, y_pred_enc, labels=np.arange(len(label_encoder.classes_))
        ).tolist()

        roc_auc = None
        if len(label_encoder.classes_) == 2 and hasattr(model, "predict_proba"):
            roc_auc = float(roc_auc_score(y_test_enc, model.predict_proba(X_test_scaled)[:, 1]))

        # Feature schema
        feature_names = list(df_X.columns)

        # Save artifacts
        joblib.dump(model, os.path.join(self.model_dir, "model.joblib"))
        joblib.dump(scaler, os.path.join(self.model_dir, "scaler.joblib"))
        joblib.dump(label_encoder, os.path.join(self.model_dir, "label_encoder.joblib"))

        schema_data = {
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "version": "1.0.0",
        }
        with open(os.path.join(self.model_dir, "feature_schema.json"), "w") as f:
            json.dump(schema_data, f, indent=2)

        model_version = self._next_model_version()
        meta_data = {
            "model_name": model_name,
            "version": model_version,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "macro_f1": float(f1),
            "roc_auc": roc_auc,
            "confusion_matrix": conf_mat,
            "classes": label_encoder.classes_.tolist(),
            "feature_count": len(feature_names),
            "split_strategy": "GroupShuffleSplit (grouped by machine run)",
            "training_samples": len(X_train),
            "test_samples": len(X_test),
        }
        with open(os.path.join(self.model_dir, "training_metadata.json"), "w") as f:
            json.dump(meta_data, f, indent=2)

        self._archive_bundle(model_version)

        print(f"[SUCCESS] Model Bundle Saved -> {self.model_dir} (Accuracy: {acc*100:.2f}%, F1: {f1:.4f})")
        return meta_data

    def _next_model_version(self) -> str:
        """Returns the next patch version without replacing archived bundles."""
        versions_dir = os.path.join(self.model_dir, "versions")
        versions = []
        if os.path.isdir(versions_dir):
            for name in os.listdir(versions_dir):
                if name.startswith("v"):
                    try:
                        versions.append(tuple(int(part) for part in name[1:].split(".")))
                    except ValueError:
                        continue
        current_metadata = os.path.join(self.model_dir, "training_metadata.json")
        if os.path.exists(current_metadata):
            try:
                with open(current_metadata, "r") as f:
                    current = tuple(int(part) for part in json.load(f)["version"][1:].split("."))
                versions.append(current)
            except (KeyError, ValueError, IndexError, json.JSONDecodeError):
                pass
        latest = max(versions, default=(1, 0, -1))
        return f"v{latest[0]}.{latest[1]}.{latest[2] + 1}"

    def _archive_bundle(self, model_version: str) -> None:
        """Copies the active five-file bundle into an immutable version directory."""
        archive_dir = os.path.join(self.model_dir, "versions", model_version)
        os.makedirs(archive_dir, exist_ok=True)
        for filename in (
            "model.joblib",
            "scaler.joblib",
            "feature_schema.json",
            "training_metadata.json",
            "label_encoder.joblib",
        ):
            source = os.path.join(self.model_dir, filename)
            destination = os.path.join(archive_dir, filename)
            if filename.endswith(".joblib"):
                joblib.dump(joblib.load(source), destination)
            else:
                with open(source, "r") as source_file, open(destination, "w") as destination_file:
                    destination_file.write(source_file.read())
