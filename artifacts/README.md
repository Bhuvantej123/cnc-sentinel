# CNC Sentinel — Model & Training Artifacts Contract

This directory defines the contract for serialized assets produced by the Machine Learning pipeline (`ml/`) and consumed by the Backend API (`backend/`).

> [!IMPORTANT]
> **Model Status: PENDING VERIFICATION**
> No production model weights or accuracy metrics are bundled in the repository at this stage. Model outputs and validation metrics remain marked as **UNAVAILABLE** until the real dataset is processed, validated, and the pipeline training script produces verified artifacts.

---

## 1. Per-Cut Feature Table Contract (`cut_features.csv`)

To decouple feature extraction from real-time API serving, the ML pipeline exports a structured per-cut summary table to `artifacts/cut_features.csv`.

### Tool Discovery Rule:
The backend **must dynamically discover tools and available cut indices** by inspecting this table (or an accompanying registry). The backend must **never hard-code invented tool records**. If `cut_features.csv` is absent, the backend reports `tools: []`.

### Required Schema:
| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `tool_id` | `string` | No | Stable tool or case identifier (e.g. `"case_01"`, `"case_02"` from NASA Milling). |
| `cut_index` | `integer` | No | 1-indexed sequential cut/pass number within the tool run ($1, 2, \dots, N$). |
| `material_id` | `string` or `integer` | Yes | Verified workpiece material (e.g. `1` for Cast Iron, `2` for Steel; null if unverified). |
| `feed_rate_mm_rev` | `float` | Yes | Verified feed rate in mm/rev (e.g. `0.25` or `0.5`; null if unverified). |
| `depth_of_cut_mm` | `float` | Yes | Verified depth of cut in mm (e.g. `0.75` or `1.5`; null if unverified). |
| `measured_wear_vb` | `float` | Yes | Optical microscope flank wear measurement ($V_B$) in mm. |
| `ae_spindle_rms` | `float` | No | Root-mean-square of spindle acoustic emission signal during this cut. |
| `ae_table_rms` | `float` | No | Root-mean-square of table acoustic emission signal during this cut. |
| `vib_spindle_rms` | `float` | No | Root-mean-square of spindle vibration acceleration. |
| `vib_table_rms` | `float` | No | Root-mean-square of table vibration acceleration. |
| `current_rms` | `float` | No | Root-mean-square of spindle motor current. |
| `vib_table_kurtosis` | `float` | No | Kurtosis of table vibration signal (spikiness indicator). |
| `vib_table_crest` | `float` | No | Crest factor of table vibration signal. |

### Handling Missing Measured Labels:
In the NASA Milling dataset, flank wear was measured intermittently.
- For cuts where wear was physically measured, `measured_wear_vb` contains the float value (e.g. `0.082`).
- For cuts where no measurement was taken, the field **must be left empty in CSV**, which loads as `NaN` in pandas and serializes to `null` in JSON. **Do not fabricate or interpolate missing ground-truth values.**

---

## 2. Model & Pipeline Artifacts

| Artifact Name | Format | Required Interface | Purpose |
| :--- | :--- | :--- | :--- |
| `tool_wear_regressor.joblib` | Binary (joblib) | Scikit-learn estimator or pipeline implementing `.predict(X)` | Predicts flank wear $V_B$ from extracted cut features. |
| `feature_scaler.joblib` | Binary (joblib) | Scikit-learn transformer implementing `.transform(X)` | Scaler fitted strictly on training data. |
| `feature_names.json` | JSON | List of string feature names matching columns in `cut_features.csv` | Guarantees feature alignment between ML and Backend. |
| `metrics_report.json` | JSON | JSON object with Leave-One-Tool-Out validation scores | Serves `/api/generalization` without recalculating at runtime. |

---

## 3. Model Loading & Validation Rules

A file existing on disk does **not** suffice to set `model_loaded = true`. The backend verifies the model artifact via `ml.inference.validate_model_artifact()`:
1. File exists and is readable.
2. File unpickles successfully into a valid scikit-learn estimator or pipeline.
3. Feature names match `feature_names.json`.
4. A test invocation with dummy inputs of the expected shape executes without exception.

Only if all 4 checks pass does the system report `model_loaded: true`. Otherwise, the backend safely falls back to `model_loaded: false` and `model_status: "UNAVAILABLE"`.

---

## 4. Git Policy

- `cut_features.csv`, `*.joblib`, `*.pkl`, `*.onnx` are excluded by root `.gitignore`.
- Only documentation and configuration schemas are checked into Git.
