"""CNC Sentinel — Model Training Script (P2: Model Training).

Trains two models against the per-cut feature table produced by the
feature-extraction pipeline:

  1. DummyRegressor (mean strategy)  — statistical baseline.
  2. Ridge regression Pipeline       — sensor-only wear model
                                       (StandardScaler -> Ridge).

Eligibility policy (per artifacts/README.md §1, Data Quality):
  * Only rows where ``training_eligible == True`` are used.
  * This enforces: is_valid_run, non-null measured_wear_vb, and
    absence of known sensor pathologies (Case 6 Run 1, Case 7 Run 4).
  * Expected supervised population: 143 cuts.

Artifact outputs (artifacts/ directory -- path approved in P1 contract):
  * artifacts/tool_wear_regressor.joblib  -- fitted sensor Pipeline
  * artifacts/feature_scaler.joblib       -- fitted StandardScaler only
  * artifacts/feature_names.json          -- ordered feature name list

Constraints:
  * Does NOT implement RUL, commit, push, or modify existing files.
  * Does NOT install packages.
  * Stops with a clear error if any required column or file is absent.

Usage:
    python ml/train_model.py [--data artifacts/cut_features.csv]
                             [--out-dir artifacts]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Guard: ensure project root is on sys.path so feature_config is importable
# regardless of the working directory the script is launched from.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Dependency check -- surface a clear blocker before importing deep stdlib deps
# ---------------------------------------------------------------------------
def _check_dependencies() -> None:
    """Raise ImportError with an actionable message if required packages are absent."""
    missing = []
    for pkg in ("numpy", "pandas", "sklearn", "joblib"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        pretty = ", ".join(missing)
        raise ImportError(
            f"Required package(s) not found: {pretty}\n"
            "Install the ML dependencies before running this script:\n"
            "    pip install -r ml/requirements.txt\n"
            "Declared versions: numpy==1.26.2, pandas==2.1.4, "
            "scikit-learn==1.3.2, joblib==1.3.2"
        )


_check_dependencies()

# ---------------------------------------------------------------------------
# Standard imports (after dependency guard)
# ---------------------------------------------------------------------------
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ml.feature_config import (  # noqa: E402
    IDENTIFIER_COLUMNS,
    QUALITY_COLUMNS,
    SENSOR_ONLY_FEATURES,
    TARGET_COLUMN,
    get_training_feature_names,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "artifacts" / "cut_features.csv"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "artifacts"

# Columns that must be present in the CSV for loading to succeed
_REQUIRED_COLUMNS: list[str] = (
    IDENTIFIER_COLUMNS
    + QUALITY_COLUMNS
    + [TARGET_COLUMN]
    + SENSOR_ONLY_FEATURES
)

# Expected supervised training population (per README §1)
_EXPECTED_TRAINING_COUNT = 143

# Ridge regularisation strength (a conservative default; tune via CV in a later phase)
_RIDGE_ALPHA = 1.0


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------

def load_and_filter(data_path: Path) -> pd.DataFrame:
    """Load ``cut_features.csv``, validate schema, and apply eligibility filter.

    Parameters
    ----------
    data_path:
        Absolute or relative path to ``artifacts/cut_features.csv``.

    Returns
    -------
    pd.DataFrame
        Subset of rows where ``training_eligible == True``.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If required columns are missing, or the training population is empty.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {data_path}\n"
            "Run the feature-extraction pipeline first:\n"
            "    python ml/prepare_data.py"
        )

    df = pd.read_csv(data_path)

    # --- Schema validation ---------------------------------------------------
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing {len(missing_cols)} required column(s): "
            f"{missing_cols}\nFound columns: {list(df.columns)}"
        )

    # --- Eligibility filter (approved policy) --------------------------------
    # ``training_eligible`` is a boolean column; handle CSV string variants.
    df["training_eligible"] = df["training_eligible"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes")
    )
    train_df = df[df["training_eligible"]].copy()

    if train_df.empty:
        raise ValueError(
            "No training-eligible rows found after applying the eligibility "
            "filter (training_eligible == True). "
            "Check that cut_features.csv was produced by the full pipeline."
        )

    # Warn (but do not block) if the count differs from the certified population.
    actual = len(train_df)
    if actual != _EXPECTED_TRAINING_COUNT:
        print(
            f"[WARNING] Expected {_EXPECTED_TRAINING_COUNT} training-eligible "
            f"rows (per artifacts/README.md §1), found {actual}. "
            "The script will proceed, but verify the feature table is current.",
            file=sys.stderr,
        )
    else:
        print(f"[INFO] Training population: {actual} cuts (matches certified count).")

    # --- Target column sanity ------------------------------------------------
    null_target = train_df[TARGET_COLUMN].isna().sum()
    if null_target > 0:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains {null_target} NaN value(s) "
            "in the training-eligible subset. "
            "The eligibility filter should guarantee non-null labels. "
            "Inspect the feature table for integrity issues."
        )

    return train_df


# ---------------------------------------------------------------------------
# Feature matrix extraction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = "sensor_only",
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X (feature matrix) and y (target vector) from the filtered DataFrame.

    Parameters
    ----------
    df:
        Training-eligible rows from ``cut_features.csv``.
    feature_set:
        Key accepted by ``get_training_feature_names``.
        Must be ``"sensor_only"`` for the wear regression model.

    Returns
    -------
    X : pd.DataFrame
        Shape ``(n_samples, n_features)`` with dtype-coerced numerics.
    y : pd.Series
        Shape ``(n_samples,)`` -- ``measured_wear_vb`` values.
    """
    feature_names = get_training_feature_names(feature_set)

    # Verify all feature columns are present (belt-and-suspenders after schema check)
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(
            f"Feature set '{feature_set}' references {len(missing)} column(s) "
            f"absent from the CSV: {missing}"
        )

    X = df[feature_names].copy().astype(float)
    y = df[TARGET_COLUMN].astype(float)

    print(
        f"[INFO] Feature matrix shape: {X.shape}  "
        f"(n_samples={X.shape[0]}, n_features={X.shape[1]})"
    )
    print(
        f"[INFO] Target '{TARGET_COLUMN}' -- "
        f"min={y.min():.4f} mm, max={y.max():.4f} mm, mean={y.mean():.4f} mm"
    )

    return X, y


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_dummy_baseline() -> DummyRegressor:
    """Return a mean-strategy DummyRegressor (unconditional statistical baseline)."""
    return DummyRegressor(strategy="mean")


def build_sensor_pipeline(alpha: float = _RIDGE_ALPHA) -> Pipeline:
    """Return a scikit-learn Pipeline: StandardScaler -> Ridge regressor.

    The scaler is fitted only on training data and stored separately
    (``feature_scaler.joblib``) so the backend can call ``.transform()``
    independently if needed.

    Parameters
    ----------
    alpha:
        Ridge regularisation strength (L2 penalty).
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, fit_intercept=True, random_state=42)),
        ]
    )


# ---------------------------------------------------------------------------
# Training & artifact persistence
# ---------------------------------------------------------------------------

def train_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
) -> dict[str, Path]:
    """Fit both models and write approved artifacts to ``out_dir``.

    Parameters
    ----------
    X:
        Feature matrix (training-eligible rows only).
    y:
        Wear target vector.
    out_dir:
        Must be the ``artifacts/`` directory (P1 approved output path).

    Returns
    -------
    dict mapping artifact name to absolute Path of the saved file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names: list[str] = list(X.columns)

    # ------------------------------------------------------------------
    # 1. DummyRegressor baseline
    # ------------------------------------------------------------------
    print("\n[TRAIN] Fitting DummyRegressor (mean strategy) baseline ...")
    dummy = build_dummy_baseline()
    dummy.fit(X, y)
    dummy_pred_mean = dummy.predict(X.iloc[:1])[0]
    print(f"[TRAIN] Dummy constant prediction: {dummy_pred_mean:.4f} mm")

    # The baseline is NOT saved as the primary artifact (it would overwrite
    # the real model slot). Kept in-memory for reporting only.
    print("[INFO] DummyRegressor fit complete (not persisted -- no approved slot).")

    # ------------------------------------------------------------------
    # 2. Sensor-only Ridge Pipeline
    # ------------------------------------------------------------------
    print("\n[TRAIN] Fitting sensor-only Pipeline (StandardScaler -> Ridge) ...")
    pipeline = build_sensor_pipeline(alpha=_RIDGE_ALPHA)
    pipeline.fit(X, y)

    train_preds = pipeline.predict(X)
    train_residuals = y.values - train_preds
    train_mae = float(np.mean(np.abs(train_residuals)))
    train_rmse = float(np.sqrt(np.mean(train_residuals ** 2)))
    print(f"[TRAIN] Ridge -- Train MAE : {train_mae:.4f} mm")
    print(f"[TRAIN] Ridge -- Train RMSE: {train_rmse:.4f} mm")
    print(
        "[NOTE] Training-set error is optimistic (no held-out validation). "
        "Generalisation metrics will be computed in a dedicated evaluation phase."
    )

    # ------------------------------------------------------------------
    # 3. Save artifacts (P1-approved paths only)
    # ------------------------------------------------------------------
    saved: dict[str, Path] = {}

    # 3a. Full pipeline -> tool_wear_regressor.joblib
    regressor_path = out_dir / "tool_wear_regressor.joblib"
    joblib.dump(pipeline, regressor_path)
    print(f"\n[SAVE] Regressor pipeline -> {regressor_path}")
    saved["tool_wear_regressor.joblib"] = regressor_path

    # 3b. Scaler only -> feature_scaler.joblib
    scaler_path = out_dir / "feature_scaler.joblib"
    scaler_only: StandardScaler = pipeline.named_steps["scaler"]
    joblib.dump(scaler_only, scaler_path)
    print(f"[SAVE] Feature scaler      -> {scaler_path}")
    saved["feature_scaler.joblib"] = scaler_path

    # 3c. Feature names -> feature_names.json
    feature_names_path = out_dir / "feature_names.json"
    with open(feature_names_path, "w", encoding="utf-8") as fh:
        json.dump(feature_names, fh, indent=2)
    print(
        f"[SAVE] Feature names       -> {feature_names_path}  "
        f"({len(feature_names)} features)"
    )
    saved["feature_names.json"] = feature_names_path

    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python ml/train_model.py",
        description="Train DummyRegressor baseline and sensor-only Ridge wear model.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_DATA_PATH,
        metavar="PATH",
        help=(
            f"Path to cut_features.csv  "
            f"[default: {_DEFAULT_DATA_PATH.relative_to(_PROJECT_ROOT)}]"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        metavar="DIR",
        help=(
            f"Directory for output artifacts  "
            f"[default: {_DEFAULT_OUT_DIR.relative_to(_PROJECT_ROOT)}]"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main training entry point."""
    args = parse_args(argv)

    data_path: Path = args.data.resolve()
    out_dir: Path = args.out_dir.resolve()

    print("=" * 70)
    print("CNC Sentinel -- Model Training (P2)")
    print("=" * 70)
    print(f"  Data path : {data_path}")
    print(f"  Output dir: {out_dir}")
    print(f"  Target    : {TARGET_COLUMN}")
    print(f"  Features  : sensor_only ({len(SENSOR_ONLY_FEATURES)} columns)")
    print("=" * 70)

    # Step 1: Load and filter
    df = load_and_filter(data_path)

    # Step 2: Build feature matrix (sensor_only -- no time counters or identifiers)
    X, y = build_feature_matrix(df, feature_set="sensor_only")

    # Step 3: Train and persist
    saved = train_and_save(X, y, out_dir)

    print("\n" + "=" * 70)
    print("Training complete. Artifacts written:")
    for name, path in saved.items():
        print(f"  {name:35s}  {path}")
    print("=" * 70)
    print(
        "\nNext steps:\n"
        "  * Run evaluation / Leave-One-Tool-Out CV in a future phase.\n"
        "  * Validate artifacts with ml.inference.validate_model_artifact().\n"
        "  * Do NOT commit *.joblib files (excluded by .gitignore per P1 policy)."
    )


if __name__ == "__main__":
    main()
