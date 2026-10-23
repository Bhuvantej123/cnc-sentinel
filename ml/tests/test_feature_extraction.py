"""CNC Sentinel — Pytest Suite for Feature Extraction and Data Quality.

Validates:
1. Synthetic sine signal with known analytical RMS, P2P, and spectral band power.
2. Variable-length signal invariance under Parseval normalization.
3. Explicit non-finite signal handling (NaN propagation, no silent zero replacement).
4. Flatline constant signal handling without division-by-zero errors.
5. Signal-quality anomaly detection (Case 2 Run 1, Case 12 Run 1, Case 6 Run 1, Case 7 Run 4).
6. Dataset integrity: 167 rows, 16 cases, 21 missing wear labels, physical run index gaps.
7. Feature configuration boundary enforcement (no leakage in sensor-only models).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.prepare_data import (
    compute_channel_features,
    evaluate_run_quality,
    SAMPLING_RATE_HZ,
)
from ml.feature_config import (
    IDENTIFIER_COLUMNS,
    PROGRESS_COLUMNS,
    SENSOR_ONLY_FEATURES,
    TARGET_COLUMN,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = REPO_ROOT / "artifacts" / "cut_features.csv"


# =========================================================================
# 1. Synthetic Signal Tests
# =========================================================================

def test_synthetic_sine_wave_statistical_and_spectral_properties():
    """Verify that compute_channel_features produces exact analytical values for a 50 Hz sine wave."""
    # 10 seconds of 50 Hz sine wave at 250 Hz sampling rate (exactly 500 integer cycles, zero leakage)
    duration_s = 10.0
    n = int(duration_s * SAMPLING_RATE_HZ)
    t = np.arange(n) / SAMPLING_RATE_HZ
    amplitude = 2.0
    frequency = 50.0
    sig = amplitude * np.sin(2.0 * np.pi * frequency * t)

    feats, err = compute_channel_features(sig, "test")
    assert err is None

    # Analytical values
    expected_mean = 0.0
    expected_rms = amplitude / np.sqrt(2.0)  # ~1.4142
    actual_discrete_p2p = float(np.max(sig) - np.min(sig))
    actual_discrete_crest = float(np.max(np.abs(sig)) / expected_rms)
    expected_variance = (amplitude ** 2) / 2.0 # 2.0

    assert abs(feats["test_mean"] - expected_mean) < 1e-12
    assert abs(feats["test_rms"] - expected_rms) < 1e-10
    assert abs(feats["test_p2p"] - actual_discrete_p2p) < 1e-10
    assert abs(feats["test_crest"] - actual_discrete_crest) < 1e-10

    # 50 Hz is inside Band 2: [30.0, 70.0) Hz
    # Band 2 should contain ~100% of signal variance (2.0)
    assert abs(feats["test_band_30_70hz"] - expected_variance) < 1e-4
    assert feats["test_band_0_30hz"] < 1e-6
    assert feats["test_band_70_125hz"] < 1e-6


def test_variable_length_energy_normalization():
    """Verify that Parseval-normalized spectral energy is invariant to signal length."""
    amplitude = 3.0
    freq = 40.0

    # Length 9,000 (36 seconds) vs Length 15,360 (61.44 seconds)
    t1 = np.arange(9000) / SAMPLING_RATE_HZ
    t2 = np.arange(15360) / SAMPLING_RATE_HZ

    sig1 = amplitude * np.cos(2.0 * np.pi * freq * t1)
    sig2 = amplitude * np.cos(2.0 * np.pi * freq * t2)

    feats1, _ = compute_channel_features(sig1, "s")
    feats2, _ = compute_channel_features(sig2, "s")

    expected_power = (amplitude ** 2) / 2.0

    assert abs(feats1["s_rms"] - feats2["s_rms"]) < 1e-3
    assert abs(feats1["s_band_30_70hz"] - expected_power) < 1e-3
    assert abs(feats2["s_band_30_70hz"] - expected_power) < 1e-3


def test_non_finite_signal_handling():
    """Verify that non-finite signals propagate NaNs and are flagged, never replaced with silent zeros."""
    sig_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    feats, err = compute_channel_features(sig_with_nan, "test")
    assert err == "NON_FINITE_SIGNAL"
    assert np.isnan(feats["test_rms"])
    assert np.isnan(feats["test_mean"])
    assert np.isnan(feats["test_band_0_30hz"])

    sig_with_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
    feats_inf, err_inf = compute_channel_features(sig_with_inf, "test")
    assert err_inf == "NON_FINITE_SIGNAL"
    assert np.isnan(feats_inf["test_rms"])


def test_flatline_constant_signal_handling():
    """Verify that constant signals (zero AC variance) evaluate safely without division-by-zero."""
    constant_sig = np.full(1000, 5.0)
    feats, err = compute_channel_features(constant_sig, "c")
    assert err is None
    assert abs(feats["c_mean"] - 5.0) < 1e-12
    assert abs(feats["c_std"] - 0.0) < 1e-12
    assert abs(feats["c_rms"] - 5.0) < 1e-12
    assert abs(feats["c_skew"] - 0.0) < 1e-12
    assert abs(feats["c_kurtosis"] - 0.0) < 1e-12
    assert abs(feats["c_band_0_30hz"] - 0.0) < 1e-12
    assert abs(feats["c_band_30_70hz"] - 0.0) < 1e-12


# =========================================================================
# 2. Quality Evaluation & Anomaly Exclusion Policy Tests
# =========================================================================

def test_quality_evaluation_rules():
    """Test policy rules for detecting variable length, spikes, flatlines, and eligibility."""
    # 1. Clean progressive run
    q_flag, is_valid, tr_elig, demo_elig = evaluate_run_quality(
        case_num=1,
        run_num=1,
        sig_lengths={"vib_table": 9000, "ae_table": 9000},
        sig_stds={"ae_table": 0.05, "ae_spindle": 0.05},
        sig_maxs={"vib_table": 2.5, "ae_table": 0.3},
        channel_errors=[],
        has_measured_wear=True,
        total_case_cuts=17,
    )
    assert q_flag == "OK"
    assert is_valid is True
    assert tr_elig is True
    assert demo_elig is True

    # 2. Variable-length run with flatline (e.g. Case 12 Run 1)
    q_flag, is_valid, tr_elig, demo_elig = evaluate_run_quality(
        case_num=12,
        run_num=1,
        sig_lengths={"vib_table": 15360, "ae_table": 15360},
        sig_stds={"ae_table": 0.00004, "ae_spindle": 0.00004},
        sig_maxs={"vib_table": 2771.0, "ae_table": 0.0002},
        channel_errors=[],
        has_measured_wear=False,
        total_case_cuts=15,
    )
    assert "VARIABLE_LENGTH" in q_flag
    assert "AE_FLATLINE" in q_flag
    assert is_valid is False
    assert tr_elig is False
    assert demo_elig is False

    # 3. Extreme voltage spike (e.g. Case 2 Run 1)
    q_flag, is_valid, tr_elig, demo_elig = evaluate_run_quality(
        case_num=2,
        run_num=1,
        sig_lengths={"vib_table": 9000, "ae_table": 9000},
        sig_stds={"ae_table": 0.05, "ae_spindle": 0.05},
        sig_maxs={"vib_table": 1e29, "ae_table": 0.3},
        channel_errors=[],
        has_measured_wear=True,
        total_case_cuts=14,
    )
    assert "EXTREME_SPIKE" in q_flag
    assert is_valid is False
    assert tr_elig is False

    # 4. Single-cut non-progressive run (e.g. Case 6 Run 1)
    q_flag, is_valid, tr_elig, demo_elig = evaluate_run_quality(
        case_num=6,
        run_num=1,
        sig_lengths={"vib_table": 9000, "ae_table": 9000},
        sig_stds={"ae_table": 0.00047, "ae_spindle": 0.05},
        sig_maxs={"vib_table": 2.5, "ae_table": 0.02},
        channel_errors=[],
        has_measured_wear=True,
        total_case_cuts=1,
    )
    assert "SINGLE_CUT_CASE" in q_flag
    assert "AE_FLATLINE" in q_flag
    assert tr_elig is False


# =========================================================================
# 3. Dataset Integration & Feature Configuration Tests
# =========================================================================

def test_feature_table_structure_and_boundaries():
    """Verify that artifacts/cut_features.csv exists and satisfies all data integrity rules."""
    if not CSV_PATH.exists():
        pytest.skip(
            "artifacts/cut_features.csv is Git-ignored and not generated on this machine; "
            "run ml/prepare_data.py to enable."
        )
    df = pd.read_csv(CSV_PATH)

    # 1. Row count and case coverage
    assert len(df) == 167
    assert len(df["tool_id"].unique()) == 16

    # 2. Physical run index preservation
    case_3_runs = df[df["tool_id"] == "case_03"]["cut_index"].tolist()
    assert case_3_runs == [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]

    case_14_runs = df[df["tool_id"] == "case_14"]["cut_index"].tolist()
    assert case_14_runs == [1, 2, 3, 4, 5, 6, 8, 9, 10]

    # 3. Missing wear labels preserved as NaN
    missing_count = df[TARGET_COLUMN].isna().sum()
    assert missing_count == 21

    # 4. Specific anomaly flagging in dataset
    c2_r1 = df[(df["tool_id"] == "case_02") & (df["cut_index"] == 1)].iloc[0]
    assert not bool(c2_r1["is_valid_run"])
    assert "EXTREME_SPIKE" in c2_r1["quality_flag"]

    c12_r1 = df[(df["tool_id"] == "case_12") & (df["cut_index"] == 1)].iloc[0]
    assert not bool(c12_r1["is_valid_run"])
    assert "VARIABLE_LENGTH" in c12_r1["quality_flag"]

    c6_r1 = df[(df["tool_id"] == "case_06") & (df["cut_index"] == 1)].iloc[0]
    assert not bool(c6_r1["training_eligible"])
    assert "AE_FLATLINE" in c6_r1["quality_flag"]

    c7_r4 = df[(df["tool_id"] == "case_07") & (df["cut_index"] == 4)].iloc[0]
    assert not bool(c7_r4["training_eligible"])
    assert "AE_FLATLINE" in c7_r4["quality_flag"]

    # 5. Feature configuration boundaries (no leakage)
    for col in IDENTIFIER_COLUMNS:
        assert col not in SENSOR_ONLY_FEATURES, f"Identifier {col} leaked into SENSOR_ONLY_FEATURES!"
    for col in PROGRESS_COLUMNS:
        assert col not in SENSOR_ONLY_FEATURES, f"Progress counter {col} leaked into SENSOR_ONLY_FEATURES!"

    # Verify that all SENSOR_ONLY_FEATURES exist in the dataframe
    for feat in SENSOR_ONLY_FEATURES:
        assert feat in df.columns, f"Configured feature {feat} missing from cut_features.csv"
