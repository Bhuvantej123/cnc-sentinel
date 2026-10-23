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

1. **Signal Length Anomaly in Case 12, Run 1**:
   - 166 runs contain exactly 9,000 points (36 seconds at 250 Hz).
   - Cut index 94 (`case=12, run=1`) contains **15,360 points**.
   - *Requirement for P2*: Feature extraction functions must compute summary statistics over the full array (or slice a uniform window e.g. `[:9000]`), and not assume `len(arr) == 9000` as a hard constraint.

2. **Non-Consecutive Run Numbering**:
   - In Case 3, recorded run identifiers are `[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]` (runs 4 and 13 were omitted during the original experiment).
   - In Case 14, run identifiers are `[1, 2, 3, 4, 5, 6, 8, 9, 10]` (run 7 omitted).
   - *Requirement for P2 & P3*: Use the actual `run` field from the struct as `cut_index` so data matches the physical records, rather than assuming a continuous 1..N index.

3. **Case 6 Premature Termination**:
   - Case 6 was terminated after cut 1 with zero wear ($V_B = 0.00$). It cannot be used alone for progressive RUL validation, but can be included as a zero-wear training sample.

4. **Wear Threshold Validity Confirmed**:
   - **15 out of 16 cases** reach and exceed $0.20\text{ mm}$ and $0.30\text{ mm}$.
   - Max wear across runs reaches $1.53\text{ mm}$.
   - This validates that a configurable default threshold in the range of **$0.20 - 0.30\text{ mm}$** is meaningful and operationally realistic for this dataset.

---

## 6. Recommendations for Task 4B (Feature Extraction by P2)

1. **Feature Schema Refinement**:
   Extract 8 primary summary statistics per sensor channel:
   - AC & DC Current: `current_ac_rms`, `current_dc_rms`
   - Spindle & Table Vibration: `vib_table_rms`, `vib_spindle_rms`, `vib_table_kurtosis`, `vib_table_crest`
   - Acoustic Emission: `ae_table_rms`, `ae_spindle_rms`
2. **Missing Label Representation**:
   Export unmeasured cuts with empty string in CSV (loading as `NaN` in pandas and `null` in JSON), exactly matching [`shared/api_contract.md`](../shared/api_contract.md).
3. **Generalization Strategy**:
   - Replicate Holdout (Train on Cases 1–8, Test on Cases 9–16).
   - Leave-One-Tool-Out (LOTO) cross-validation across all 15 progressive cases.
