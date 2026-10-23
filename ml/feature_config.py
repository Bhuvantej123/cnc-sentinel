"""CNC Sentinel — Feature Configuration & Model Input Boundaries.

Explicitly partitions features to prevent temporal and identifier leakage:
- Identifiers and run counters are strictly excluded from sensor models.
- Sensor features and operating conditions form the primary sensor model input.
- A progress/time baseline is defined separately for benchmark comparison.
- Quality and eligibility columns control sample filtering.
"""

from typing import List

# Identifiers and raw sequence counters (NEVER used as regression model inputs)
IDENTIFIER_COLUMNS: List[str] = [
    "tool_id",
    "cut_index",
    "case_num",
    "run_num",
]

# Physical ground-truth target
TARGET_COLUMN: str = "measured_wear_vb"

# Verified cutting parameters (controllable operating conditions)
OPERATING_CONDITION_COLUMNS: List[str] = [
    "material_id",
    "feed_rate_mm_rev",
    "depth_of_cut_mm",
]

# Temporal progress counter (used solely in the progress baseline model)
PROGRESS_COLUMNS: List[str] = [
    "cutting_time_min",
]

# Quality and eligibility metadata columns
QUALITY_COLUMNS: List[str] = [
    "quality_flag",
    "is_valid_run",
    "training_eligible",
    "demo_eligible",
]

# Base sensor channel prefixes
SENSOR_PREFIXES: List[str] = [
    "smc_ac",
    "smc_dc",
    "vib_table",
    "vib_spindle",
    "ae_table",
    "ae_spindle",
]

# Standard feature metrics extracted per channel
FEATURE_METRICS: List[str] = [
    "mean",
    "std",
    "rms",
    "min",
    "max",
    "p2p",
    "skew",
    "kurtosis",
    "crest",
    "band_0_30hz",
    "band_30_70hz",
    "band_70_125hz",
]

# Complete list of all 72 channel features + 1 composite current RMS
SENSOR_FEATURE_COLUMNS: List[str] = [
    f"{prefix}_{metric}" for prefix in SENSOR_PREFIXES for metric in FEATURE_METRICS
] + ["current_rms"]

# Model Feature Sets:
# 1. Primary Sensor Model: Strictly sensor features + operating conditions (NO identifiers or time counters)
SENSOR_ONLY_FEATURES: List[str] = SENSOR_FEATURE_COLUMNS + OPERATING_CONDITION_COLUMNS

# 2. Progress Baseline Model: Uses solely cutting time and operating conditions for benchmark comparison
PROGRESS_BASELINE_FEATURES: List[str] = PROGRESS_COLUMNS + OPERATING_CONDITION_COLUMNS

# 3. Combined Features: Sensor features + operating conditions + cumulative time
COMBINED_FEATURES: List[str] = SENSOR_ONLY_FEATURES + PROGRESS_COLUMNS


def get_training_feature_names(feature_set: str = "sensor_only") -> List[str]:
    """Retrieve verified ordered feature names for a given model architecture."""
    if feature_set == "sensor_only":
        return list(SENSOR_ONLY_FEATURES)
    elif feature_set == "progress_baseline":
        return list(PROGRESS_BASELINE_FEATURES)
    elif feature_set == "combined":
        return list(COMBINED_FEATURES)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}. Choose 'sensor_only', 'progress_baseline', or 'combined'.")
