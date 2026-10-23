"""CNC Sentinel — NASA Milling Dataset Feature Extraction Pipeline.

Reads data/raw/mill.mat, extracts robust time-domain and frequency-domain
features across all six sensor channels, evaluates signal quality, and exports
the per-cut feature table to artifacts/cut_features.csv.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats

# NOTE: The 250 Hz sampling rate is derived from published NASA PCoE / UC Berkeley
# literature and experimental setup documentation. It is NOT stored as a field inside mill.mat.
SAMPLING_RATE_HZ: float = 250.0

# Non-overlapping frequency bands within the 125 Hz Nyquist range.
# Band 1 starts at 0.5 Hz to omit sub-rotational DC leakage after demeaning.
# Intervals are half-open [f_low, f_high), with the final band closed at Nyquist [70.0, 125.0].
SPECTRAL_BANDS: List[Tuple[str, float, float]] = [
    ("0_30hz", 0.5, 30.0),
    ("30_70hz", 30.0, 70.0),
    ("70_125hz", 70.0, 125.0),
]

# Six physical sensor channels available in mill.mat
SENSOR_CHANNELS: List[Tuple[str, str]] = [
    ("smcAC", "smc_ac"),
    ("smcDC", "smc_dc"),
    ("vib_table", "vib_table"),
    ("vib_spindle", "vib_spindle"),
    ("AE_table", "ae_table"),
    ("AE_spindle", "ae_spindle"),
]

# Physical voltage thresholds for anomaly detection
VOLTAGE_SPIKE_LIMIT_V: float = 50.0
AE_FLATLINE_STD_LIMIT_V: float = 0.002


def compute_channel_features(signal: np.ndarray, prefix: str) -> Tuple[Dict[str, float], Optional[str]]:
    """Compute statistical and spectral features for a single sensor channel.

    Features:
    - Statistical moments: mean, std (ddof=1), rms, min, max, p2p
    - Distribution shapes: skewness, kurtosis (Fisher, zero for Gaussian)
    - Waveform metric: crest factor (guarded against zero-RMS division)
    - Spectral energy: Demeaned, Parseval-normalized one-sided FFT band energy

    Returns:
    - (features_dict, error_flag_or_None)
    """
    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)

    # 1. Explicit handling of empty or non-finite values (NO silent zero replacement)
    if n == 0:
        nan_feats = {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_rms": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_p2p": np.nan,
            f"{prefix}_skew": np.nan,
            f"{prefix}_kurtosis": np.nan,
            f"{prefix}_crest": np.nan,
        }
        for b_name, _, _ in SPECTRAL_BANDS:
            nan_feats[f"{prefix}_band_{b_name}"] = np.nan
        return nan_feats, "EMPTY_SIGNAL"

    if not np.all(np.isfinite(sig)):
        nan_feats = {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_rms": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_p2p": np.nan,
            f"{prefix}_skew": np.nan,
            f"{prefix}_kurtosis": np.nan,
            f"{prefix}_crest": np.nan,
        }
        for b_name, _, _ in SPECTRAL_BANDS:
            nan_feats[f"{prefix}_band_{b_name}"] = np.nan
        return nan_feats, "NON_FINITE_SIGNAL"

    # 2. Time-domain statistical moments
    mean_val = float(np.mean(sig))
    std_val = float(np.std(sig, ddof=1)) if n > 1 else 0.0
    rms_val = float(np.sqrt(np.mean(sig**2)))
    min_val = float(np.min(sig))
    max_val = float(np.max(sig))
    p2p_val = max_val - min_val

    # 3. Skewness and Kurtosis (guarded for constant/flatline signals)
    if std_val > 1e-12:
        skew_val = float(stats.skew(sig, bias=False))
        kurt_val = float(stats.kurtosis(sig, bias=False))
    else:
        skew_val = 0.0
        kurt_val = 0.0

    # 4. Crest factor: peak amplitude / RMS
    peak_val = float(np.max(np.abs(sig)))
    crest_val = float(peak_val / rms_val) if rms_val > 1e-12 else 0.0

    features = {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_rms": rms_val,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_p2p": p2p_val,
        f"{prefix}_skew": skew_val,
        f"{prefix}_kurtosis": kurt_val,
        f"{prefix}_crest": crest_val,
    }

    # 5. Spectral band energy: Demean before FFT and apply Parseval normalization
    sig_detrended = sig - mean_val
    rfft_vals = np.fft.rfft(sig_detrended)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLING_RATE_HZ)

    # Parseval-normalized one-sided power spectrum such that sum(power_spectrum) == Var(sig)
    power_spectrum = (np.abs(rfft_vals) ** 2) / (n ** 2)
    if n % 2 == 0:
        power_spectrum[1:-1] *= 2.0
    else:
        power_spectrum[1:] *= 2.0

    for idx, (b_name, f_low, f_high) in enumerate(SPECTRAL_BANDS):
        # Use half-open intervals [f_low, f_high), closed on final Nyquist band
        if idx == len(SPECTRAL_BANDS) - 1:
            mask = (freqs >= f_low) & (freqs <= f_high)
        else:
            mask = (freqs >= f_low) & (freqs < f_high)

        band_energy = float(np.sum(power_spectrum[mask])) if np.any(mask) else 0.0
        features[f"{prefix}_band_{b_name}"] = band_energy

    return features, None


def extract_flank_wear(vb_raw: Any) -> Optional[float]:
    """Safely extract measured flank wear in mm from VB struct field.

    Returns None for unmeasured cuts. Never interpolates or fabricates.
    """
    if isinstance(vb_raw, (int, float, np.number)):
        v = float(vb_raw)
        return v if not np.isnan(v) else None
    elif isinstance(vb_raw, (np.ndarray, list)):
        if len(vb_raw) > 0:
            v = float(vb_raw[0])
            return v if not np.isnan(v) else None
    return None


def evaluate_run_quality(
    case_num: int,
    run_num: int,
    sig_lengths: Dict[str, int],
    sig_stds: Dict[str, float],
    sig_maxs: Dict[str, float],
    channel_errors: List[str],
    has_measured_wear: bool,
    total_case_cuts: int,
) -> Tuple[str, bool, bool, bool]:
    """Evaluate data quality and determine model training and demo eligibility.

    Returns:
    - quality_flag (str): Semicolon-delimited descriptive flags or 'OK'
    - is_valid_run (bool): False for severe corruptions/overflows
    - training_eligible (bool): False if unmeasured, corrupted, flatlined, or single-cut
    - demo_eligible (bool): True for verified progressive benchmark cases
    """
    flags: List[str] = []
    is_valid_run = True

    # Check for channel-level non-finite errors
    if channel_errors:
        flags.extend(channel_errors)
        is_valid_run = False

    # Check for variable length
    for ch, length in sig_lengths.items():
        if length != 9000:
            flags.append(f"VARIABLE_LENGTH_{length}")
            is_valid_run = False
            break

    # Check for extreme voltage magnitude spikes (e.g. Case 2 Run 1 has 1e29+ V overflow)
    extreme_channels = [ch for ch, m in sig_maxs.items() if m > VOLTAGE_SPIKE_LIMIT_V]
    if extreme_channels:
        flags.append(f"EXTREME_SPIKE_{','.join(extreme_channels)}")
        is_valid_run = False

    # Check for flatlined acoustic emission channels
    ae_flatlines = []
    if sig_stds.get("ae_table", 1.0) < AE_FLATLINE_STD_LIMIT_V:
        ae_flatlines.append("ae_table")
    if sig_stds.get("ae_spindle", 1.0) < AE_FLATLINE_STD_LIMIT_V:
        ae_flatlines.append("ae_spindle")

    if ae_flatlines:
        flags.append(f"AE_FLATLINE_{','.join(ae_flatlines)}")

    # Check for single-cut cases (Case 6 has only 1 cut, no progressive wear degradation)
    if total_case_cuts <= 1:
        flags.append("SINGLE_CUT_CASE")

    # Format quality flag
    quality_str = "; ".join(flags) if flags else "OK"

    # Training Eligibility Policy:
    # Must be valid run, must have measured wear, must not have flatlines or single-cut runs
    has_flatline = any("AE_FLATLINE" in f for f in flags)
    training_eligible = (
        is_valid_run
        and has_measured_wear
        and not has_flatline
        and total_case_cuts > 1
    )

    # Demo Eligibility Policy:
    # Benchmark progressive cases without sensor dropouts or extreme anomalies
    # (Cases 1, 3, 9, 10, 11 are ideal candidates)
    clean_progressive_cases = {1, 3, 9, 10, 11}
    demo_eligible = (case_num in clean_progressive_cases) and is_valid_run and (quality_str == "OK")

    return quality_str, is_valid_run, training_eligible, demo_eligible


def extract_dataset_features(mat_file_path: Path) -> Tuple[pd.DataFrame, List[Tuple[int, str]]]:
    """Parse mill.mat and extract all cut records and engineered features."""
    if not mat_file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {mat_file_path}")

    mat = sio.loadmat(str(mat_file_path), struct_as_record=False, squeeze_me=True)
    if "mill" not in mat:
        raise KeyError("Expected top-level variable 'mill' not found in MAT file.")

    mill_data = mat["mill"]
    rows: List[Dict[str, Any]] = []
    skipped_records: List[Tuple[int, str]] = []

    # Pre-count total cuts per case to detect isolated/single-cut runs
    case_cut_counts: Dict[int, int] = {}
    for run in mill_data:
        c = int(getattr(run, "case"))
        case_cut_counts[c] = case_cut_counts.get(c, 0) + 1

    for idx, run in enumerate(mill_data):
        try:
            case_id = int(getattr(run, "case"))
            run_id = int(getattr(run, "run"))
            tool_id = f"case_{case_id:02d}"
            cut_index = run_id

            material_id = int(getattr(run, "material"))
            feed_rate = float(getattr(run, "feed"))
            depth_of_cut = float(getattr(run, "DOC"))
            cutting_time = float(getattr(run, "time"))
            vb_val = extract_flank_wear(getattr(run, "VB"))

            row: Dict[str, Any] = {
                "tool_id": tool_id,
                "cut_index": cut_index,
                "case_num": case_id,
                "run_num": run_id,
                "material_id": material_id,
                "feed_rate_mm_rev": feed_rate,
                "depth_of_cut_mm": depth_of_cut,
                "cutting_time_min": cutting_time,
                "measured_wear_vb": vb_val,
            }

            sig_lengths: Dict[str, int] = {}
            sig_stds: Dict[str, float] = {}
            sig_maxs: Dict[str, float] = {}
            channel_errors: List[str] = []

            # Extract features across all 6 sensor channels
            for mat_field, prefix in SENSOR_CHANNELS:
                sig = getattr(run, mat_field)
                sig_lengths[prefix] = len(sig) if hasattr(sig, "__len__") else 0
                sig_maxs[prefix] = float(np.max(sig)) if len(sig) > 0 and np.all(np.isfinite(sig)) else float("inf")
                sig_stds[prefix] = float(np.std(sig)) if len(sig) > 0 and np.all(np.isfinite(sig)) else 0.0

                ch_feats, err = compute_channel_features(sig, prefix)
                if err:
                    channel_errors.append(f"{prefix}_{err}")
                row.update(ch_feats)

            # Composite overall current RMS for contract backward-compatibility
            smc_ac_rms = row.get("smc_ac_rms", np.nan)
            smc_dc_rms = row.get("smc_dc_rms", np.nan)
            if np.isfinite(smc_ac_rms) and np.isfinite(smc_dc_rms):
                row["current_rms"] = float(np.sqrt(smc_ac_rms ** 2 + smc_dc_rms ** 2))
            else:
                row["current_rms"] = np.nan

            # Evaluate run quality and exclusion flags
            total_cuts_in_case = case_cut_counts.get(case_id, 0)
            q_flag, is_valid, tr_elig, demo_elig = evaluate_run_quality(
                case_num=case_id,
                run_num=run_id,
                sig_lengths=sig_lengths,
                sig_stds=sig_stds,
                sig_maxs=sig_maxs,
                channel_errors=channel_errors,
                has_measured_wear=(vb_val is not None),
                total_case_cuts=total_cuts_in_case,
            )

            row["quality_flag"] = q_flag
            row["is_valid_run"] = is_valid
            row["training_eligible"] = tr_elig
            row["demo_eligible"] = demo_elig

            rows.append(row)
        except Exception as e:
            skipped_records.append((idx, str(e)))

    df = pd.DataFrame(rows)
    return df, skipped_records


def run_pipeline(mat_path: Path, output_path: Path) -> pd.DataFrame:
    """Execute feature extraction and export artifacts/cut_features.csv."""
    t0 = time.time()
    print(f"Loading raw dataset from {mat_path}...")
    df, skipped = extract_dataset_features(mat_path)

    if skipped:
        print(f"WARNING: Skipped {len(skipped)} malformed records: {skipped}")
    else:
        print(f"Successfully processed all {len(df)} cut records with 0 skipped.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    elapsed = time.time() - t0

    missing_vb_count = df["measured_wear_vb"].isna().sum()
    measured_vb_count = len(df) - missing_vb_count
    training_eligible_count = df["training_eligible"].sum()
    flagged_anomalies_count = (df["quality_flag"] != "OK").sum()

    print(f"Exported per-cut feature table to: {output_path}")
    print(f"  Total Runs: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Measured Wear Cuts: {measured_vb_count}")
    print(f"  Missing Wear Cuts (preserved as NaN): {missing_vb_count}")
    print(f"  Flagged Quality Anomaly Cuts: {flagged_anomalies_count}")
    print(f"  Training Eligible Cuts: {training_eligible_count}")
    print(f"  Total Processing Time: {elapsed:.2f}s")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNC Sentinel NASA Milling Feature Extractor")
    parser.add_argument(
        "--mat-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw" / "mill.mat",
        help="Path to data/raw/mill.mat",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "cut_features.csv",
        help="Path to export artifacts/cut_features.csv",
    )
    args = parser.parse_args()

    run_pipeline(args.mat_path, args.output_path)
