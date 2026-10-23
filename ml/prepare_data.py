"""CNC Sentinel — NASA Milling Dataset Feature Extraction Pipeline.

Reads data/raw/mill.mat, extracts robust time-domain and frequency-domain
features across all six sensor channels, and exports the per-cut feature table
to artifacts/cut_features.csv in strict accordance with the artifact contract.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats

# Sampling frequency of the NASA Milling Dataset
SAMPLING_RATE_HZ = 250.0

# Frequency bands within Nyquist limit (0 - 125 Hz)
SPECTRAL_BANDS = [
    ("0_30hz", 0.0, 30.0),
    ("30_70hz", 30.0, 70.0),
    ("70_125hz", 70.0, 125.0),
]

# Six available physical sensor channels in mill.mat
SENSOR_CHANNELS: List[Tuple[str, str]] = [
    ("smcAC", "smc_ac"),
    ("smcDC", "smc_dc"),
    ("vib_table", "vib_table"),
    ("vib_spindle", "vib_spindle"),
    ("AE_table", "ae_table"),
    ("AE_spindle", "ae_spindle"),
]


def compute_channel_features(signal: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute statistical and spectral features for a single sensor channel signal.

    Handles variable-length arrays and degenerate constant signals safely.
    """
    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)

    # Fallback for degenerate or empty signals
    if n == 0 or not np.all(np.isfinite(sig)):
        degenerate_dict = {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_rms": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p2p": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_kurtosis": 0.0,
            f"{prefix}_crest": 0.0,
        }
        for b_name, _, _ in SPECTRAL_BANDS:
            degenerate_dict[f"{prefix}_band_{b_name}"] = 0.0
        return degenerate_dict

    # Time-domain statistical moments
    mean_val = float(np.mean(sig))
    std_val = float(np.std(sig, ddof=1)) if n > 1 else 0.0
    rms_val = float(np.sqrt(np.mean(sig**2)))
    min_val = float(np.min(sig))
    max_val = float(np.max(sig))
    p2p_val = max_val - min_val

    # Skewness and Kurtosis (Fisher's definition: normal distribution kurtosis = 0)
    if std_val > 1e-12:
        skew_val = float(stats.skew(sig, bias=False))
        kurt_val = float(stats.kurtosis(sig, bias=False))
    else:
        skew_val = 0.0
        kurt_val = 0.0

    # Crest factor: peak amplitude / RMS
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

    # Frequency-domain: Real FFT and band energy
    rfft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLING_RATE_HZ)
    power_spectrum = (np.abs(rfft_vals) ** 2) / n

    for b_name, f_low, f_high in SPECTRAL_BANDS:
        mask = (freqs >= f_low) & (freqs <= f_high)
        band_energy = float(np.sum(power_spectrum[mask])) if np.any(mask) else 0.0
        features[f"{prefix}_band_{b_name}"] = band_energy

    return features


def extract_flank_wear(vb_raw: Any) -> float | None:
    """Safely extract measured flank wear in mm from the VB struct field.

    Returns None for unmeasured / missing cuts. Never interpolates.
    """
    if isinstance(vb_raw, (int, float, np.number)):
        v = float(vb_raw)
        return v if not np.isnan(v) else None
    elif isinstance(vb_raw, (np.ndarray, list)):
        if len(vb_raw) > 0:
            v = float(vb_raw[0])
            return v if not np.isnan(v) else None
    return None


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

            # Extract features across all 6 sensor channels
            for mat_field, prefix in SENSOR_CHANNELS:
                sig = getattr(run, mat_field)
                ch_feats = compute_channel_features(sig, prefix)
                row.update(ch_feats)

            # Composite overall current RMS for contract backward-compatibility
            row["current_rms"] = float(np.sqrt(row["smc_ac_rms"] ** 2 + row["smc_dc_rms"] ** 2))

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

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV (missing values written as empty string, which pandas reads back as NaN)
    df.to_csv(output_path, index=False)
    elapsed = time.time() - t0

    missing_vb_count = df["measured_wear_vb"].isna().sum()
    measured_vb_count = len(df) - missing_vb_count

    print(f"Exported per-cut feature table to: {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Measured Wear Records: {measured_vb_count}")
    print(f"  Missing Wear Records (preserved as NaN): {missing_vb_count}")
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
