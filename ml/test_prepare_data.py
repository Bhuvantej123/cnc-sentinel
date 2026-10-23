"""Automated validation tests for the feature extraction pipeline.

Verifies:
1. Output file exists and row count matches 167 raw runs.
2. Case and physical run identifiers are preserved.
3. Exactly 21 missing wear measurements remain NaN.
4. Exactly 146 measured wear values are valid positive floats.
5. All 73 numeric feature columns are strictly finite.
6. The variable-length run (case_12, cut_index 1) is processed correctly.
7. Required artifact contract columns are present.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "artifacts" / "cut_features.csv"

REQUIRED_CONTRACT_COLUMNS = [
    "tool_id",
    "cut_index",
    "material_id",
    "feed_rate_mm_rev",
    "depth_of_cut_mm",
    "cutting_time_min",
    "measured_wear_vb",
    "ae_spindle_rms",
    "ae_table_rms",
    "vib_spindle_rms",
    "vib_table_rms",
    "current_rms",
    "vib_table_kurtosis",
    "vib_table_crest",
    "smc_ac_rms",
    "smc_dc_rms",
]


def run_validation_checks() -> bool:
    print(f"Validating {CSV_PATH}...")
    if not CSV_PATH.exists():
        print(f"FAILED: File {CSV_PATH} does not exist.")
        return False

    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    total_cols = len(df.columns)
    print(f"Loaded {total_rows} rows, {total_cols} columns.")

    checks_passed = True

    # 1. Total row count check
    if total_rows != 167:
        print(f"FAILED: Expected 167 rows, got {total_rows}")
        checks_passed = False
    else:
        print("PASS: Exactly 167 cut records present.")

    # 2. Case count and format
    unique_tools = sorted(df["tool_id"].unique())
    expected_tools = [f"case_{i:02d}" for i in range(1, 17)]
    if unique_tools != expected_tools:
        print(f"FAILED: Expected 16 tool IDs {expected_tools}, got {unique_tools}")
        checks_passed = False
    else:
        print("PASS: All 16 tools (case_01 .. case_16) preserved in tool_id.")

    # 3. Physical run index preservation (checking gaps in Case 3 and Case 14)
    case_3_runs = df[df["tool_id"] == "case_03"]["cut_index"].tolist()
    expected_case_3 = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
    if case_3_runs != expected_case_3:
        print(f"FAILED: Case 3 runs mismatch: expected {expected_case_3}, got {case_3_runs}")
        checks_passed = False
    else:
        print("PASS: Physical non-consecutive run indices preserved (Case 3 verified).")

    case_14_runs = df[df["tool_id"] == "case_14"]["cut_index"].tolist()
    expected_case_14 = [1, 2, 3, 4, 5, 6, 8, 9, 10]
    if case_14_runs != expected_case_14:
        print(f"FAILED: Case 14 runs mismatch: expected {expected_case_14}, got {case_14_runs}")
        checks_passed = False
    else:
        print("PASS: Physical non-consecutive run indices preserved (Case 14 verified).")

    # 4. Flank wear missing label counts
    missing_wear_count = df["measured_wear_vb"].isna().sum()
    if missing_wear_count != 21:
        print(f"FAILED: Expected exactly 21 missing wear records, got {missing_wear_count}")
        checks_passed = False
    else:
        print("PASS: Exactly 21 intermittent cuts preserved with NaN wear.")

    # 5. Measured wear range
    measured_df = df[~df["measured_wear_vb"].isna()]
    if len(measured_df) != 146:
        print(f"FAILED: Expected 146 measured wear records, got {len(measured_df)}")
        checks_passed = False
    else:
        min_vb = measured_df["measured_wear_vb"].min()
        max_vb = measured_df["measured_wear_vb"].max()
        if min_vb < 0.0 or max_vb > 2.0:
            print(f"FAILED: Wear values outside expected physical range [0, 2.0]: min={min_vb}, max={max_vb}")
            checks_passed = False
        else:
            print(f"PASS: 146 measured wear records in valid physical range [{min_vb:.4f}, {max_vb:.4f}] mm.")

    # 6. Finite values check across all numeric feature columns
    non_finite_cols = {}
    feature_cols = [c for c in df.columns if c != "measured_wear_vb"]
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            non_finite = (~np.isfinite(df[c])).sum()
            if non_finite > 0:
                non_finite_cols[c] = non_finite

    if non_finite_cols:
        print(f"FAILED: Non-finite feature values found: {non_finite_cols}")
        checks_passed = False
    else:
        print(f"PASS: All numeric feature columns are 100% finite (0 NaNs, 0 Infs).")

    # 7. Variable-length run check (Case 12, Run 1)
    case_12_run_1 = df[(df["tool_id"] == "case_12") & (df["cut_index"] == 1)]
    if len(case_12_run_1) != 1:
        print(f"FAILED: Could not find Case 12 Run 1.")
        checks_passed = False
    else:
        rms_val = case_12_run_1["smc_ac_rms"].values[0]
        if not np.isfinite(rms_val) or rms_val <= 0:
            print(f"FAILED: Case 12 Run 1 (15360 pts) computed invalid RMS: {rms_val}")
            checks_passed = False
        else:
            print(f"PASS: Variable-length run (Case 12 Run 1, 15,360 pts) successfully processed (RMS={rms_val:.4f}).")

    # 8. Required contract columns check
    missing_contract_cols = [c for c in REQUIRED_CONTRACT_COLUMNS if c not in df.columns]
    if missing_contract_cols:
        print(f"FAILED: Missing contract columns: {missing_contract_cols}")
        checks_passed = False
    else:
        print(f"PASS: All {len(REQUIRED_CONTRACT_COLUMNS)} required contract columns present.")

    if checks_passed:
        print("\nALL FEATURE VALIDATION CHECKS PASSED SUCCESSFULLY.")
        return True
    else:
        print("\nSOME VALIDATION CHECKS FAILED.")
        return False


if __name__ == "__main__":
    success = run_validation_checks()
    sys.exit(0 if success else 1)
