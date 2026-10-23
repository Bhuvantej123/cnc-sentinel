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
| `tool_id` | `string` | No | Stable tool identifier (`"case_01"` to `"case_16"`). |
| `cut_index` | `integer` | No | Physical run number within the tool case ($1 \dots N$, preserving non-consecutive runs). |
| `material_id` | `integer` | No | Verified workpiece material (`1` = Cast Iron, `2` = Steel). |
| `feed_rate_mm_rev` | `float` | No | Verified feed rate in mm/rev (`0.25` or `0.5`). |
| `depth_of_cut_mm` | `float` | No | Verified depth of cut in mm (`0.75` or `1.5`). |
| `cutting_time_min` | `float` | No | Cumulative machining duration in minutes. |
| `measured_wear_vb` | `float` | Yes | Optical microscope flank wear measurement ($V_B$) in mm. Null for unmeasured cuts. |
| `quality_flag` | `string` | No | Semicolon-delimited quality codes (`"OK"`, `"EXTREME_SPIKE_*"`, `"VARIABLE_LENGTH_*"`, `"AE_FLATLINE_*"`, `"SINGLE_CUT_CASE"`). |
| `is_valid_run` | `boolean` | No | `False` for severe bit-corruption or transmission overflows (e.g. Case 2 Run 1). |
| `training_eligible` | `boolean` | No | `True` strictly for uncorrupted runs with measured wear and valid progressive sensor signals. |
| `demo_eligible` | `boolean` | No | `True` for clean progressive benchmark cases (Cases 1, 3, 9, 10, 11). |
| `current_rms` | `float` | No | Engineering composite: $\sqrt{\text{smc\_ac\_rms}^2 + \text{smc\_dc\_rms}^2}$. Provides an aggregate electrical load proxy for backward-compatibility with single-current contracts, without claiming physical electrical equivalence to true total current. |
| `smc_ac_*`, `smc_dc_*` | `float` | No | AC and DC motor current features (12 features each). |
| `vib_table_*`, `vib_spindle_*` | `float` | No | Table and spindle vibration features (12 features each). |
| `ae_table_*`, `ae_spindle_*` | `float` | No | Table and spindle acoustic emission features (12 features each). |

#### Extracted Metrics Per Channel (12 features × 6 channels = 72 features):
For each channel prefix (`smc_ac`, `smc_dc`, `vib_table`, `vib_spindle`, `ae_table`, `ae_spindle`):
- `*_mean`: Arithmetic signal mean ($\mu$).
- `*_std`: Sample standard deviation ($\sigma$, `ddof=1`).
- `*_rms`: Root-mean-square amplitude ($\sqrt{\frac{1}{N}\sum x_i^2}$).
- `*_min`: Minimum signal amplitude.
- `*_max`: Maximum signal amplitude.
- `*_p2p`: Peak-to-peak amplitude range ($\max - \min$).
- `*_skew`: Sample skewness (`stats.skew`, `bias=False`).
- `*_kurtosis`: Excess kurtosis (`stats.kurtosis`, `bias=False`, Gaussian = 0.0).
- `*_crest`: Crest factor ($\max(|x|) / \text{rms}$, guarded against zero division).
- `*_band_0_30hz`: Parseval-normalized one-sided power in $[0.5, 30.0)\text{ Hz}$.
- `*_band_30_70hz`: Parseval-normalized one-sided power in $[30.0, 70.0)\text{ Hz}$.
- `*_band_70_125hz`: Parseval-normalized one-sided power in $[70.0, 125.0]\text{ Hz}$.

### Spectral Processing & Sampling Details:
- **Sampling Rate Assumption**: 250 Hz (derived from published NASA PCoE literature; not stored in `mill.mat`).
- **Mean Removal (Demeaning)**: Signal mean is subtracted prior to FFT computation to prevent DC contamination of the low-frequency band.
- **Spectral Bands**: Non-overlapping intervals within the 125 Hz Nyquist frequency:
  - `band_0_30hz`: $[0.5, 30.0)\text{ Hz}$ (omits sub-0.5 Hz residual DC offset)
  - `band_30_70hz`: $[30.0, 70.0)\text{ Hz}$ (structural vibration & tooth-passing harmonics)
  - `band_70_125hz`: $[70.0, 125.0]\text{ Hz}$ (high-frequency friction & acoustic emissions)
- **Parseval Energy Normalization**: Scaled such that $\sum \text{power\_spectrum} = \text{Var}(x)$, making band energy estimates invariant to signal duration.

### Data Quality & Training Exclusion Policy:
1. **Preservation of Raw Records**: All 167 raw runs remain present in `artifacts/cut_features.csv` (no silent deletions).
2. **Distinction Between `is_valid_run` and `training_eligible`**:
   - **`is_valid_run` (Signal Fidelity)**: Evaluates whether the raw recording is free from fatal ADC/transmission corruption ($> 50.0\text{ V}$) or non-standard array lengths.
     - **Case 6 Run 1 is valid** (`is_valid_run = True`): 9,000 points, normal finite voltages.
     - **Case 7 Run 4 is valid** (`is_valid_run = True`): 9,000 points, normal voltages on vibration, current, and spindle AE.
     - **Case 2 Run 1 is invalid** (`is_valid_run = False`): Catastrophic numerical overflow ($> 10^{19}\text{ V}$ up to $1.58 \times 10^{29}\text{ V}$).
     - **Case 12 Run 1 is invalid** (`is_valid_run = False`): Abnormal 15,360-point length, flatlined AE, and current channel cap.
   - **`training_eligible` (Model Training Suitability)**: Evaluates whether a run should be included in supervised regression training.
     - Requires `is_valid_run == True`.
     - Requires non-null target label (`measured_wear_vb` is not NaN).
     - **Excludes Case 6 Run 1** (`training_eligible = False`): Case 6 was terminated after cut 1 ($V_B = 0.00\text{ mm}$), offering no progressive degradation curve; table AE is also at noise floor ($0.0207\text{ V}$).
     - **Excludes Case 7 Run 4** (`training_eligible = False`): Table AE channel experienced an isolated dropout to sensor noise floor ($0.0216\text{ V}$ vs. normal $0.10 - 0.15\text{ V}$), which would inject a misleading zero-energy anomaly into an otherwise progressive multi-cut run.
3. **Certified Population Counts**:
   - Total records: **167 cuts**
   - Valid signal records (`is_valid_run == True`): **165 cuts**
   - Measured wear records: **146 cuts**
   - Unmeasured wear records (NaN): **21 cuts**
   - Supervised training population (`training_eligible == True`): **143 cuts** ($146\text{ measured} - 1\text{ [Case 2 Run 1]} - 1\text{ [Case 6 Run 1]} - 1\text{ [Case 7 Run 4]} = 143$).
   - Demo candidate cases (`demo_eligible == True`): Cases 1, 3, 9, 10, and 11 (**73 cuts**).

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
