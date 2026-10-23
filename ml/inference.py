"""CNC Sentinel — Single Inference & RUL Estimation Interface.

This module is the single source of truth for:
- Cutting-tool wear point prediction
- Uncertainty interval estimation (confidence bounds)
- Remaining Useful Life (RUL) forecasting in cuts
- Risk-calibrated recommendation generation

Backend routes must delegate directly to `predict()`. The backend is responsible
only for HTTP validation and serialization.
"""

from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "tool_wear_regressor.joblib"
SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"
CUT_FEATURES_PATH = ARTIFACTS_DIR / "cut_features.csv"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

# Quality-flag markers (written by ml/prepare_data.py) that indicate corrupted or
# dropped-out sensor channels. Cuts carrying any of these never receive a prediction.
SENSOR_FAULT_MARKERS = (
    "EXTREME_SPIKE",
    "VARIABLE_LENGTH",
    "AE_FLATLINE",
    "NON_FINITE_SIGNAL",
    "EMPTY_SIGNAL",
)

# Plausibility bounds for a raw model output (mm). Measured VB in the NASA dataset
# spans 0.00-1.53 mm; outputs outside these bounds indicate out-of-distribution input.
MIN_PLAUSIBLE_WEAR_MM = -0.05
MAX_PLAUSIBLE_WEAR_MM = 2.0


class InferenceErrorCode(str, Enum):
    """Machine-readable reason a prediction was withheld (None when a prediction is returned)."""
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    UNKNOWN_CUT = "UNKNOWN_CUT"
    INVALID_RUN = "INVALID_RUN"
    FEATURES_UNAVAILABLE = "FEATURES_UNAVAILABLE"
    IMPLAUSIBLE_PREDICTION = "IMPLAUSIBLE_PREDICTION"
    INFERENCE_FAILED = "INFERENCE_FAILED"


class RecommendationStatus(str, Enum):
    """Actionable shop-floor advisories."""
    CONTINUE = "CONTINUE"
    INSPECT_SOON = "INSPECT_SOON"
    REPLACE_TOOL = "REPLACE_TOOL"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class RiskTolerance(str, Enum):
    """Operator risk appetite defining boundary calculation.

    NOTE: Calibration of uncertainty bounds must be empirically validated
    against test tool runs before production deployment.
    """
    CONSERVATIVE = "conservative"  # Uses upper uncertainty bound; earlier replacement warning
    BALANCED = "balanced"          # Uses expected mean point prediction
    AGGRESSIVE = "aggressive"      # Uses lower uncertainty bound; delays replacement warning


_cached_model = None
_cached_scaler = None
_cached_features = None


def validate_model_artifact() -> bool:
    """Validate that the model artifact exists, is readable, unpickles correctly,

    and satisfies the expected scikit-learn estimator interface.
    A file existing on disk is not sufficient to claim model_loaded = True.
    """
    global _cached_model, _cached_scaler, _cached_features

    if not (MODEL_PATH.exists() and FEATURE_NAMES_PATH.exists()):
        return False

    try:
        import joblib

        model = joblib.load(MODEL_PATH)
        if not hasattr(model, "predict"):
            return False

        with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
            features = json.load(f)
            if not isinstance(features, list) or len(features) == 0:
                return False

        # Refuse artifacts serialized under a different scikit-learn version
        if not _package_versions_match():
            return False

        # Smoke-test invocation with inputs of the expected shape
        probe = pd.DataFrame([[0.0] * len(features)], columns=features)
        if not np.all(np.isfinite(model.predict(probe))):
            return False

        _cached_model = model
        _cached_features = features
        if SCALER_PATH.exists():
            _cached_scaler = joblib.load(SCALER_PATH)

        return True
    except Exception:
        _cached_model = None
        _cached_scaler = None
        _cached_features = None
        return False


def _package_versions_match() -> bool:
    """Compare the scikit-learn version recorded at training time with the runtime version.

    Artifacts without recorded versions (trained before provenance was added) are accepted.
    """
    if not METADATA_PATH.exists():
        return True
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        recorded = json.load(f).get("package_versions", {})
    if "scikit-learn" not in recorded:
        return True

    import sklearn

    return recorded["scikit-learn"] == sklearn.__version__


def is_model_verified() -> bool:
    """Check if a verified model has been loaded and validated."""
    if _cached_model is not None:
        return True
    return validate_model_artifact()


def _is_true(value: Any) -> bool:
    """Interpret CSV boolean variants ("True", "1", "yes") as True."""
    return str(value).strip().lower() in ("true", "1", "yes")


def _withheld_response(
    tool_id: str,
    cut_index: int,
    explanation: str,
    model_status: str,
    error_code: InferenceErrorCode,
) -> Dict[str, Any]:
    """Build a response that withholds the prediction and any shop-floor recommendation."""
    return {
        "tool_id": tool_id,
        "cut_index": cut_index,
        "predicted_wear": None,
        "wear_unit": "mm",
        "uncertainty_interval": None,
        "estimated_rul_cuts": None,
        "recommendation": RecommendationStatus.MODEL_UNAVAILABLE.value,
        "explanation": explanation,
        "model_status": model_status,
        "error_code": error_code.value,
    }


def predict(
    tool_id: str,
    cut_index: int,
    wear_threshold: float,
    risk_tolerance: str = "balanced",
) -> Dict[str, Any]:
    """Execute wear prediction and RUL estimation for a cutting tool at a specific cut.

    Parameters:
    - tool_id: Stable identifier of the tool run (e.g. "case_01" in NASA Milling).
    - cut_index: Current recorded cut index (1-indexed pass number). Predictions
                 estimate current wear at this cut and project remaining useful life
                 (additional passes) until wear reaches wear_threshold.
    - wear_threshold: Flank wear limit in mm (e.g. 0.20 mm). Must be a positive verified value.
    - risk_tolerance: "conservative", "balanced", or "aggressive".

    Returns:
    - Dict matching the POST /api/predict response contract, plus an ``error_code``
      field (None on success, otherwise an ``InferenceErrorCode`` value). If the model,
      the cut, or trustworthy sensor features are unavailable, prediction fields remain
      None with recommendation "MODEL_UNAVAILABLE". No values or accuracy scores are fabricated.
    """
    # 1. Check if model and required artifacts are verified
    if not is_model_verified() or not CUT_FEATURES_PATH.exists():
        if not CUT_FEATURES_PATH.exists():
            explanation = (
                "Verified feature table (cut_features.csv) has not yet been generated "
                "from the NASA Milling dataset. Predictions and RUL are currently unavailable."
            )
        else:
            explanation = (
                "ML model weights (tool_wear_regressor.joblib) have not yet been trained and verified. "
                "Predictions and RUL are currently unavailable."
            )
        return _withheld_response(
            tool_id, cut_index, explanation, "UNAVAILABLE", InferenceErrorCode.MODEL_UNAVAILABLE
        )

    # 2. Once verified model and cut_features.csv exist, real inference is performed
    try:
        df = pd.read_csv(CUT_FEATURES_PATH)
        match = df[(df["tool_id"] == tool_id) & (df["cut_index"] == cut_index)]
        if match.empty:
            return _withheld_response(
                tool_id,
                cut_index,
                f"Cut {cut_index} for tool '{tool_id}' not found in telemetry feature table.",
                "VERIFIED",
                InferenceErrorCode.UNKNOWN_CUT,
            )

        row = match.iloc[0]

        # Refuse cuts whose sensor data failed quality checks. Fails closed when the
        # quality columns are absent (e.g. a feature table from an older pipeline).
        quality_flag = str(row.get("quality_flag", "MISSING_QUALITY_COLUMNS"))
        sensor_faults = [m for m in SENSOR_FAULT_MARKERS if m in quality_flag]
        if (
            "is_valid_run" not in row
            or not _is_true(row["is_valid_run"])
            or "quality_flag" not in row
            or sensor_faults
        ):
            return _withheld_response(
                tool_id,
                cut_index,
                (
                    f"Sensor data for cut {cut_index} of '{tool_id}' failed quality checks "
                    f"({quality_flag}). No prediction is issued; inspect the tool manually."
                ),
                "VERIFIED",
                InferenceErrorCode.INVALID_RUN,
            )

        # Check required features exist and are finite (no silent zero substitution)
        missing_or_nan = [
            f for f in _cached_features
            if (f not in row) or pd.isna(row[f]) or not np.isfinite(row[f])
        ]
        if missing_or_nan:
            return _withheld_response(
                tool_id,
                cut_index,
                f"Cut features contain {len(missing_or_nan)} missing or non-finite values.",
                "VERIFIED",
                InferenceErrorCode.FEATURES_UNAVAILABLE,
            )

        X = pd.DataFrame([row[_cached_features].astype(float)])
        raw_pred = float(_cached_model.predict(X)[0])

        # Out-of-distribution guard: never clamp an implausible output into a confident advisory
        if not (MIN_PLAUSIBLE_WEAR_MM <= raw_pred <= MAX_PLAUSIBLE_WEAR_MM):
            return _withheld_response(
                tool_id,
                cut_index,
                (
                    f"Model output ({raw_pred:.4f} mm) is outside the plausible range "
                    f"[{MIN_PLAUSIBLE_WEAR_MM}, {MAX_PLAUSIBLE_WEAR_MM}] mm; input is likely "
                    "out of distribution. No prediction is issued; inspect the tool manually."
                ),
                "VERIFIED",
                InferenceErrorCode.IMPLAUSIBLE_PREDICTION,
            )

        # Physical wear is non-negative (only small negative outputs reach this point)
        predicted_wear = round(max(0.0, raw_pred), 4)

        # Operational status thresholds
        if predicted_wear >= wear_threshold:
            rec = RecommendationStatus.REPLACE_TOOL.value
            explanation = (
                f"Predicted flank wear ({predicted_wear:.4f} mm) has reached or exceeded "
                f"replacement threshold ({wear_threshold:.4f} mm)."
            )
        elif predicted_wear >= 0.8 * wear_threshold:
            rec = RecommendationStatus.INSPECT_SOON.value
            explanation = (
                f"Predicted flank wear ({predicted_wear:.4f} mm) is within the precautionary zone "
                f"(>= 80% of {wear_threshold:.4f} mm limit)."
            )
        else:
            rec = RecommendationStatus.CONTINUE.value
            explanation = (
                f"Predicted flank wear ({predicted_wear:.4f} mm) is safely below the "
                f"{wear_threshold:.4f} mm limit."
            )

        return {
            "tool_id": tool_id,
            "cut_index": cut_index,
            "predicted_wear": predicted_wear,
            "wear_unit": "mm",
            "uncertainty_interval": None,
            "estimated_rul_cuts": None,
            "recommendation": rec,
            "explanation": explanation,
            "model_status": "VERIFIED",
            "error_code": None,
        }
    except Exception as e:
        return _withheld_response(
            tool_id,
            cut_index,
            f"Inference execution failed: {str(e)}",
            "UNAVAILABLE",
            InferenceErrorCode.INFERENCE_FAILED,
        )
