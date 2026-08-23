# VibraGuard ML & Feature Pipeline

## Data Split Strategy
To avoid data leakage from adjacent overlapping vibration windows, dataset splitting is strictly grouped by **machine/run/file ID** rather than random row splitting.

## Artifact Serialization Standard
Every trained model deployment bundle contains:
1. `model.joblib` — Trained Classifier (Random Forest, SVM, etc.)
2. `scaler.joblib` — Standard/Robust Scaler fitted on training split only
3. `feature_schema.json` — Exact ordered list of expected features and types
4. `training_metadata.json` — Hyperparameters, split strategy, metrics (F1, Accuracy, Precision, Recall)
5. `label_encoder.joblib` — Target label mapper

## Fixed VMD Acceptance Check
Phase 2 uses `vmdpy.VMD` with explicit fixed `K` and `alpha` values. The engine returns
the full IMF matrix for downstream feature extraction, downsampled waveform previews for
the API, per-IMF statistics, and the relative reconstruction error:

`||signal - sum(IMFs)||_2 / ||signal||_2`

The known-signal verification uses the first 2,048 samples of
`ml/data/sample/healthy_bearing.csv`, with `K=4` and `alpha=2000`. A valid decomposition
must return four IMFs and a relative reconstruction error below `0.15`.

## Adaptive VMD Search
Phase 3 evaluates the fixed baseline (`K=5`, `alpha=2000`) plus every candidate in
the requested inclusive `K` range and a deterministic geometric alpha grid from
`500` through `10000`. Lower fitness is preferred. The current fitness is:

`0.4 * reconstruction_error + 0.3 * mean_spectral_entropy - 0.2 * normalized_mean_kurtosis + 0.1 * mode_overlap_penalty`

The normalized mean kurtosis is clipped to `[0, 1]` after dividing the mean IMF
kurtosis by `10`. Mode overlap is penalized when adjacent sorted center frequencies
are less than `20 Hz` apart. The optimizer returns the full candidate trajectory
and retains the baseline whenever no candidate improves its measured fitness.

## Training Evaluation and Versioning
Training uses grouped holdout evaluation and searches for a deterministic split where
every class appears in both partitions. This prevents reported macro metrics from
being inflated by a test set missing a class. Accuracy, macro precision, macro recall,
macro F1, confusion matrix, and binary ROC-AUC (when probabilities are available) are
recorded from real holdout predictions.

The active five-file bundle remains in `models/` for inference. Each run is also copied
to `models/versions/<version>/`, with patch versions incremented from the latest saved
bundle so previous model metadata and artifacts remain available.

## Explainability
The SHAP layer provides local attribution for one prediction and global mean absolute
attribution across feature rows supplied by the caller. IMF features are grouped into
`imf_1`, `imf_2`, and so on; non-IMF features are grouped as `raw_signal`. These values
describe which features contributed most strongly to the model prediction and do not
prove physical causation.
