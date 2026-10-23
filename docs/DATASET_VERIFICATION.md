# NASA Milling Dataset — Verification & Real-Data Inspection Report

**Project**: CNC Sentinel (AI-Powered CNC Tool Wear Prediction & RUL Estimation)  
**Author**: Senior Software Architect / Integration Lead  
**Audience**: P1 (Team Lead & Integrator), P2 (ML Engineer), P3 (Backend Developer)  
**Status**: Task 3 Complete — Dataset Verified from Official Documentation; Awaiting Local Archive Download by P1.

---

## Executive Summary

Before committing to feature extraction or model training, the team required formal verification of the primary candidate dataset. This report documents the verified technical specification of the **NASA Milling Dataset** from official documentation and publications (NASA Ames Prognostics Center of Excellence and UC Berkeley BEST Laboratory).

> [!IMPORTANT]
> **Verification Status Distinction**:
> - **Verified from Documentation**: File structure, sensor channels, sampling frequency, 16 test cases, 167 runs, cutting condition matrix, intermittent $V_B$ labels, and 21 missing values.
> - **Verified from Local Files**: None. The local repository does **not** currently contain raw dataset files (strictly adhering to the rule prohibiting automated unapproved downloads).
> - **Assumptions & Configurable Parameters**: Flank wear threshold (e.g. 0.20 mm) is documented as **unverified and configurable** until confirmed against the distribution of physical cuts. No model accuracy or predictions are fabricated.

---

## 1. Official Dataset Provenance & Acquisition Guide for P1

The official benchmark was generated at UC Berkeley (A. Agogino & K. Goebel, 2007) and distributed by NASA Ames Research Center:

- **Official Repository Entry**: [NASA Open Data Portal — Milling Wear](https://data.nasa.gov/dataset/milling-wear)
- **Direct Resource Archive**: `https://data.nasa.gov/docs/legacy/mill.zip` (containing `mill.mat` and `readme.txt`)
- **NASA PCoE Archive**: [Prognostics Center of Excellence Data Set Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

### Action Required by P1:
1. Download `mill.zip` from the official link above.
2. Unpack `mill.mat` into `data/raw/` (target path: `data/raw/mill.mat`).
3. Note: `data/raw/` is ignored by Git, ensuring repository size remains clean.

---

## 2. Technical Specifications (Verified from Official Documentation)

### 2.1 File Format & Internal Hierarchy
- **Primary File**: `mill.mat` (MATLAB structured binary format, readable in Python via `scipy.io.loadmat('data/raw/mill.mat', struct_as_record=False, squeeze_me=True)`).
- **Structure**: A $1 \times 167$ array of structures (`mill`), where each element represents a single cutting pass (run).
- **Fields in Each Struct Element**:
  - `case`: Case number ($1, 2, \dots, 16$).
  - `run`: Sequential cut index within the case ($1, \dots, N$).
  - `VB`: Flank wear measurement in millimeters ($V_B$).
  - `time`: Cutting duration of this pass.
  - `DOC`: Depth of cut (mm).
  - `feed`: Feed rate (mm/rev or mm/min).
  - `material`: Workpiece material flag (`1` = Cast Iron, `2` = Steel).
  - `smcAC`: Spindle motor AC current time-series.
  - `smcDC`: Spindle motor DC current time-series.
  - `vib_table`: Accelerometer vibration at table.
  - `vib_spindle`: Accelerometer vibration at spindle.
  - `AE_table`: Acoustic emission at table.
  - `AE_spindle`: Acoustic emission at spindle.

### 2.2 Cases and Cutting Runs
- **Total Cases**: 16 tool cases.
- **Total Cuts / Runs**: 167 runs across all 16 cases.
- **Run Distribution**: Varies per case (typically 6 to 17 runs per tool case, running progressively until severe wear or experiment termination).

### 2.3 Sensor Channels & Sampling Details
- **Sensor Count**: 6 continuous signal streams:
  1. `smcAC`: Spindle Motor AC Current
  2. `smcDC`: Spindle Motor DC Current
  3. `vib_table`: Accelerometer Vibration (Machine Table)
  4. `vib_spindle`: Accelerometer Vibration (Spindle Housing)
  5. `AE_table`: Acoustic Emission (Table)
  6. `AE_spindle`: Acoustic Emission (Spindle)
- **Sampling Frequency**: **250 Hz**.
- **Data Points per Signal**: **9,000 points** per channel per cut (representing 36 seconds of recorded machining per pass).

### 2.4 Flank Wear ($V_B$) Ground Truth & Missing Labels
- **Target Variable**: $V_B$ (width of the flank wear land).
- **Unit**: **Millimeters (mm)** (typical observed range in NASA records: $0.00\text{ mm}$ to $\sim 0.45\text{ mm}$).
- **Intermittent Measurement**: Wear was physically measured under an optical microscope between cuts at irregular intervals.
  - **Measured Cuts**: Exactly **146 runs** contain a valid numerical $V_B$ measurement.
  - **Unmeasured / Missing Cuts**: Exactly **21 runs** have no physical microscope measurement recorded (`VB` is blank / `NaN`).
  - **Handling Rule**: The ML pipeline must preserve these 21 runs as `NaN` in pandas and `null` in JSON. No fake values, linear interpolation, or forward-fill may be applied without explicit domain justification.

### 2.5 Cutting Conditions & Generalization Support

Across the 16 cases, cutting speed is held constant ($200\text{ m/min}$), while a $2 \times 2 \times 2$ factorial design was executed:
- **Depth of Cut (DOC)**: $0.75\text{ mm}$ vs. $1.5\text{ mm}$
- **Feed Rate**: $0.25\text{ mm/rev}$ ($206.5\text{ mm/min}$) vs. $0.5\text{ mm/rev}$ ($413\text{ mm/min}$)
- **Material**: Cast Iron (coded `1`) vs. Stainless Steel (coded `2`)

| Case Number | Material | Feed Rate (mm/rev) | DOC (mm) | Design Role |
| :--- | :--- | :--- | :--- | :--- |
| **Case 1** | Cast Iron (1) | 0.50 | 1.50 | Baseline Condition A |
| **Case 2** | Cast Iron (1) | 0.50 | 0.75 | Baseline Condition B |
| **Case 3** | Cast Iron (1) | 0.25 | 0.75 | Baseline Condition C |
| **Case 4** | Cast Iron (1) | 0.25 | 1.50 | Baseline Condition D |
| **Case 5** | Steel (2) | 0.50 | 1.50 | Baseline Condition E |
| **Case 6** | Steel (2) | 0.25 | 1.50 | Baseline Condition F |
| **Case 7** | Steel (2) | 0.25 | 0.75 | Baseline Condition G |
| **Case 8** | Steel (2) | 0.50 | 0.75 | Baseline Condition H |
| **Cases 9–16** | Varied | Varied | Varied | **Direct Replicates** of Cases 1–8 under identical parameter sets |

#### Defensible Generalization Evaluation:
This structure strongly supports two defensible validation strategies:
1. **Replicate Holdout (LOTO Pair Evaluation)**: Train on Cases 1–8, evaluate on unseen replicate tools (Cases 9–16), or execute Leave-One-Tool-Out cross validation across the 16 cases.
2. **Cross-Condition Generalization**: Train on Cast Iron (Cases 1–4, 9–12), test zero-shot generalization on Steel (Cases 5–8, 13–16); or hold out high feed rate tools.

---

## 3. Wear Threshold Policy: Configurable & Pending Confirmation

> [!WARNING]
> **The 0.20 mm wear threshold is NOT confirmed as an absolute physical constant.**
> While ISO 3685 specifies standard tool life criteria ($V_B = 0.30\text{ mm}$ for uniform wear or $0.20\text{ mm}$ for fine finishing), the exact threshold depends on the specific workpiece quality criteria and the physical maximum wear observed in the tool run.
> 
> In CNC Sentinel:
> 1. `wear_threshold` in `POST /api/predict` **must remain a required, operator-configurable input parameter** with a reasonable default (`0.20 mm`).
> 2. The threshold must never be hard-coded into the ML training logic as a fixed constant.
> 3. Once `mill.mat` is loaded by P2, the maximum observed $V_B$ per tool case must be checked to confirm that typical runs reach or exceed this threshold before calculating RUL.

---

## 4. Gap Analysis: Current Artifact Contract vs. Verified Dataset

Comparing the verified NASA Milling data against [`artifacts/README.md`](../artifacts/README.md) and [`shared/api_contract.md`](../shared/api_contract.md):

| Feature / Dimension | Verified NASA Data | Current Contract (`cut_features.csv`) | Assessment & Alignment |
| :--- | :--- | :--- | :--- |
| **Tool Identifier** | `case` ($1 \dots 16$) | `tool_id` (e.g. `"case_01"`) | **Compatible**: Formatted as `"case_01"` to `"case_16"`. |
| **Cut Identifier** | `run` ($1 \dots N$) | `cut_index` ($1 \dots N$) | **Compatible**: Exactly 1-indexed pass number. |
| **Operating Conditions** | `material` (1, 2), `feed`, `DOC` | `material_id`, `feed_rate_mm_rev`, `depth_of_cut_mm` | **Compatible**: Exact match. |
| **Wear Ground Truth** | `VB` in mm (146 measured, 21 NaN) | `measured_wear_vb` in mm (null for missing) | **Compatible**: Strict alignment. |
| **Motor Current Sensors** | Both AC (`smcAC`) and DC (`smcDC`) | Single `current_rms` column in contract | **Recommended Refinement**: In Task 4, P2 should extract both `current_ac_rms` and `current_dc_rms` (or both along with overall current) to capture both spindle load and drive current dynamics. |
| **Acoustic & Vibration Sensors** | 4 channels: `AE_table`, `AE_spindle`, `vib_table`, `vib_spindle` | `ae_spindle_rms`, `ae_table_rms`, `vib_spindle_rms`, `vib_table_rms`, `vib_table_kurtosis`, `vib_table_crest` | **Compatible**: Contract covers all 4 channels and standard spectral/temporal moments. |

*Contract Conclusion*: The current feature table schema cleanly supports the NASA Milling dataset without breaking changes to `shared/api_contract.md`.

---

## 5. Summary of Decisions Required from P1

1. **Obtain Raw Dataset**: P1 to download `mill.zip` from [data.nasa.gov](https://data.nasa.gov/docs/legacy/mill.zip) and extract `mill.mat` into `data/raw/mill.mat`.
2. **Current Sensor Columns**: Approve refining the feature extraction contract to include separate `current_ac_rms` and `current_dc_rms` features.
3. **Generalization Benchmark Scheme**: Confirm whether Leave-One-Tool-Out (LOTO across all 16 cases) or Replicate Holdout (Train 1–8, Test 9–16) is the primary story for the pitch deck.
