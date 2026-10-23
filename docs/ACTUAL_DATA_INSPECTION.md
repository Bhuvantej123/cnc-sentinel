# Actual NASA Milling Dataset — Direct Binary Inspection Report

**Project**: CNC Sentinel  
**Inspection Date**: 2026-10-23  
**Target File**: `data/raw/mill.mat`  
**Inspector**: Senior Software Architect / Lead Integrator  
**Status**: Real Dataset Verified Directly from Disk via `scipy.io.loadmat`

---

## 1. File Inventory & Storage Validation

- **File Path**: `data/raw/mill.mat`
- **File Size**: 72,575,144 bytes (~72.58 MB)
- **Format / Header**: `MATLAB 5.0 MAT-file, Platform: PCWIN, Created on: Tue Nov 27 15:59:45 2007` (Version 1.0)
- **Compatibility**: Compatible with standard `scipy.io.loadmat(..., struct_as_record=False, squeeze_me=True)`. Not MATLAB v7.3 HDF5; no extra HDF5 library required.
- **Git Status**: Confirmed that `data/raw/mill.mat` is **strictly Git-ignored** by the root `.gitignore`. Working tree remains clean; no data is staged or committed.

---

## 2. Actual Variable Structure & Field Inventory

Top-level MAT variables found: `['mill']`

`mill` is a 1-D numpy array of structured objects of length **167** (`shape=(167,)`). Each entry corresponds to an individual machining cut.

### Verified Struct Fields per Cut:

| Field Name | Data Type in Python | Observed Range / Shape | Description |
| :--- | :--- | :--- | :--- |
| `case` | `int` | $1$ to $16$ | Experimental case number. |
| `run` | `int` | $1$ to $23$ | Sequential cut number within the case. |
| `VB` | `float` or empty array | $0.0000$ to $1.5300\text{ mm}$ (or `NaN`/empty) | Optical microscope flank wear measurement. |
| `time` | `float` / `int` | $0.0$ to $105.0$ (minutes) | Cumulative cutting duration within that tool case. |
| `DOC` | `float` | $0.75$ or $1.50\text{ mm}$ | Depth of cut. Constant within each case. |
| `feed` | `float` | $0.25$ or $0.50\text{ mm/rev}$ | Feed rate. Constant within each case. |
| `material` | `int` | $1$ or $2$ | Workpiece material flag: `1` = Cast Iron, `2` = Steel. |
| `smcAC` | `ndarray` (float64) | $(9000,)$ or $(15360,)$ | AC Spindle Motor Current signal. |
| `smcDC` | `ndarray` (float64) | $(9000,)$ or $(15360,)$ | DC Spindle Motor Current signal. |
| `vib_table` | `ndarray` (float64) | $(9000,)$ or $(15360,)$ | Table Accelerometer Vibration signal. |
| `vib_spindle` | `ndarray` (float64) | $(9000,)$ or $(15360,)$ | Spindle Accelerometer Vibration signal. |
| `AE_table` | `ndarray` (float64) | $(9000,)$ or $(15360,)$ | Table Acoustic Emission signal. |
| `AE_spindle` | `ndarray` (float64) | $(9000,)$ or $(15360,)$ | Spindle Acoustic Emission signal. |

> [!CRITICAL]
> **Absence of Force Measurements**:
> The dataset **does not contain force channels** (no dynamometer measurements, no $F_x$, $F_y$, $F_z$). Sensor signals are exclusively **current**, **vibration**, and **acoustic emission**. We must never claim or assume force data is available in the feature extractor or UI.

---

## 3. Ground Truth Counts: Parsed Facts vs. Prior Assumptions

| Dimension | Documented Claim | Actually Parsed Count | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Total Machining Runs** | 167 | **167** | Exact match. |
| **Unique Tool Cases** | 16 | **16** | Exact match (`case` values: 1–16). |
| **Measured Flank Wear ($V_B$) Runs** | 146 | **146** | Exactly 146 runs have a physical microscope measurement. |
| **Missing Flank Wear ($V_B$) Runs** | 21 | **21** | Exactly 21 runs have missing/NaN wear. |
| **Minimum Measured $V_B$** | ~0.00 mm | **0.0000 mm** | Sharp new inserts start at 0 wear. |
| **Maximum Measured $V_B$** | ~0.45 mm | **1.5300 mm** | Actual wear extends substantially higher than typical literature summaries. |
| **Mean Measured $V_B$** | — | **0.3376 mm** | Median is 0.2850 mm. |
| **Signal Lengths** | "All 9,000 points" | **166 runs @ 9,000; 1 run @ 15,360** | **Key finding**: Run index 94 (`case=12, run=1`) has 15,360 points across all 6 sensor channels. |

---

## 4. Case-by-Case Tool Wear & Progression Profiles

Direct parsing reveals the complete wear and cut distribution for all 16 cases:

| Case | Total Cuts | Measured Cuts | Min $V_B$ (mm) | Max $V_B$ (mm) | Material | Feed (mm/rev) | DOC (mm) | Exceeds 0.20 mm? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 17 | 13 | 0.00 | 0.50 | 1 (Cast Iron) | 0.50 | 1.50 | Yes |
| **2** | 14 | 13 | 0.08 | 0.55 | 1 (Cast Iron) | 0.50 | 0.75 | Yes |
| **3** | 14 | 14 | 0.00 | 0.55 | 1 (Cast Iron) | 0.25 | 0.75 | Yes |
| **4** | 7 | 7 | 0.08 | 0.49 | 1 (Cast Iron) | 0.25 | 1.50 | Yes |
| **5** | 6 | 6 | 0.00 | 0.74 | 2 (Steel) | 0.50 | 1.50 | Yes |
| **6** | 1 | 1 | 0.00 | 0.00 | 2 (Steel) | 0.25 | 1.50 | **No (Only 1 cut)** |
| **7** | 8 | 7 | 0.00 | 0.46 | 2 (Steel) | 0.25 | 0.75 | Yes |
| **8** | 6 | 5 | 0.00 | 0.62 | 2 (Steel) | 0.50 | 0.75 | Yes |
| **9** | 9 | 9 | 0.00 | 0.81 | 1 (Cast Iron) | 0.50 | 1.50 | Yes |
| **10** | 10 | 10 | 0.00 | 0.70 | 1 (Cast Iron) | 0.25 | 1.50 | Yes |
| **11** | 23 | 20 | 0.00 | 0.76 | 1 (Cast Iron) | 0.25 | 0.75 | Yes |
| **12** | 15 | 12 | 0.05 | 0.65 | 1 (Cast Iron) | 0.50 | 0.75 | Yes |
| **13** | 15 | 13 | 0.10 | 1.53 | 2 (Steel) | 0.25 | 0.75 | Yes |
| **14** | 9 | 7 | 0.09 | 1.14 | 2 (Steel) | 0.50 | 0.75 | Yes |
| **15** | 7 | 6 | 0.15 | 0.70 | 2 (Steel) | 0.25 | 1.50 | Yes |
| **16** | 6 | 3 | 0.24 | 0.62 | 2 (Steel) | 0.50 | 1.50 | Yes |

### Breakdown of the 21 Missing $V_B$ Cuts by Case:
- **Case 1**: Runs 2, 3, 5, 16 (4 missing)
- **Case 2**: Run 6 (1 missing)
- **Case 7**: Run 8 (1 missing)
- **Case 8**: Run 4 (1 missing)
- **Case 11**: Runs 7, 14, 17 (3 missing)
- **Case 12**: Runs 1, 4, 10 (3 missing)
- **Case 13**: Runs 1, 2 (2 missing)
- **Case 14**: Runs 1, 5 (2 missing)
- **Case 15**: Run 1 (1 missing)
- **Case 16**: Runs 1, 2, 4 (3 missing)
- **Cases 3, 4, 5, 6, 9, 10**: Zero missing wear measurements (100% complete labels).

---

## 5. Critical Technical Anomalies Identified for Pipeline Safety

1. **Extreme Sensor Overflow in Case 2, Run 1 (Index 17)**:
   - Contains numerical values exceeding $10^{19}\text{ V}$ up to $1.58 \times 10^{29}\text{ V}$ across `vib_table`, `vib_spindle`, `smcAC`, and `smcDC`, and $10^{34}\text{ V}$ in `AE_table`.
   - Represents bit-corruption or ADC transmission overflow in the original laboratory recording.
   - *Action*: Flagged as `EXTREME_SPIKE_vib_table,vib_spindle,smcAC,smcDC`. Marked `is_valid_run = False` and `training_eligible = False`.

2. **Signal Length & Sensor Failure in Case 12, Run 1 (Index 94)**:
   - 166 runs contain exactly 9,000 points. Cut index 94 (`case=12, run=1`) contains **15,360 points**.
   - `AE_table` and `AE_spindle` are flatlined at $0.0000\text{ V}$ (std $\approx 0.00004\text{ V}$). Vibration and current channels exhibit synthetic maximum values of exactly $2771.1897\text{ V}$. Wear measurement is missing (`NaN`).
   - *Action*: Flagged as `VARIABLE_LENGTH_15360; AE_FLATLINE_ae_table,ae_spindle; EXTREME_SPIKE_vib_table,vib_spindle,smcAC,smcDC`. Marked `is_valid_run = False` and `training_eligible = False`.

3. **Table Acoustic Emission Dropout in Case 7, Run 4 (Index 119)**:
   - `AE_table` drops abruptly to sensor noise floor (mean $0.0216\text{ V}$, std $0.00060\text{ V}$), while adjacent runs (Run 3: mean $0.1057\text{ V}$; Run 5: mean $0.1464\text{ V}$) and all other channels (`vib_table`, `AE_spindle`, `smcAC`, `smcDC`) behave normally.
   - Indicates a temporary table AE transducer or cabling disconnection during that cut.
   - *Action*: Flagged as `AE_FLATLINE_ae_table`. Excluded from model training (`training_eligible = False`).

4. **Premature Run Termination in Case 6, Run 1 (Index 115)**:
   - Case 6 was terminated after cut 1 ($V_B = 0.00\text{ mm}$), providing no wear progression. Table AE channel is also at noise floor ($0.0207\text{ V}$, std $0.00047\text{ V}$).
   - *Action*: Flagged as `SINGLE_CUT_CASE; AE_FLATLINE_ae_table`. Excluded from model training (`training_eligible = False`).

5. **Non-Consecutive Run Numbering**:
   - Case 3 run identifiers: `[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]` (runs 4 and 13 omitted).
   - Case 14 run identifiers: `[1, 2, 3, 4, 5, 6, 8, 9, 10]` (run 7 omitted).
   - *Action*: Physical `run` identifiers are preserved in `cut_index` and `run_num`.

---

## 6. Documented Exclusion Policy & Certified Populations

To ensure rigorous data science without silently deleting raw data:
- **Total Population**: 167 raw runs remain present in `artifacts/cut_features.csv`.
- **Valid Signal Population (`is_valid_run == True`)**: **165 cuts** (excludes Case 2 Run 1 and Case 12 Run 1).
- **Training Population (`training_eligible == True`)**: **143 cuts**.
  - Excludes 21 cuts with unmeasured wear (`measured_wear_vb is NaN`).
  - Excludes 2 severely corrupt cuts (Case 2 Run 1 and Case 12 Run 1).
  - Excludes 1 flatlined dropout cut with measured wear (Case 7 Run 4).
  - Excludes 1 single-cut isolated case (Case 6 Run 1).
- **Demo Selection (`demo_eligible == True`)**: Cases 1, 3, 9, 10, 11 provide clean progressive wear without channel dropouts or sequence anomalies.

---

## 7. Python Environment & Dependency Observations

The local development environment uses **Python 3.13.6**:
- **Installed Packages**: `numpy==2.3.5`, `scipy==1.17.1`, `scikit-learn==1.5.2`, `pandas==2.2.3`, `fastapi==0.128.8`, `pydantic==2.13.5`, `pytest==8.4.2`.
- **Observation on Requirements Files**:
  The initial `requirements.txt` files specified Python 3.10/3.11/3.12 versions (`numpy==1.26.2`, `scikit-learn==1.3.2`). Python 3.13 changed CPython internal C-APIs; prebuilt binary wheels for older scikit-learn 1.3.2 are not available for Python 3.13.
- **Recommendation**:
  For developers using Python 3.13+, use compatible modern pins:
  - `numpy>=2.0.0`
  - `scipy>=1.14.0`
  - `scikit-learn>=1.5.0`
  - `pandas>=2.2.0`
  For developers using Python 3.10–3.12, the original pinned versions in `requirements.txt` remain supported. Do not execute automated package upgrades across contributor machines without team consensus.
