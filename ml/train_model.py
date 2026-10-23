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
                             [--features combined|sensor_only]
                             [--no-tune --alpha ALPHA]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

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
            "    pip install -r ml/requirements.txt"
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
    COMBINED_FEATURES,
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

# Group column that defines tool identity
_GROUP_COLUMN = "tool_id"

# Columns that must be present in the CSV for loading to succeed
_REQUIRED_COLUMNS: list[str] = (
    IDENTIFIER_COLUMNS
    + QUALITY_COLUMNS
    + [TARGET_COLUMN]
    + COMBINED_FEATURES
)

# Expected supervised training population (per README §1)
_EXPECTED_TRAINING_COUNT = 143

# Baseline Ridge regularisation strength (1.0 for backward-compatible sensor baseline)
_RIDGE_ALPHA = 1.0
_OPTIMAL_COMBINED_ALPHA = 100.0
_ALPHA_GRID: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]

# Packages whose versions determine whether serialized artifacts can be reloaded
_PROVENANCE_PACKAGES: tuple[str, ...] = ("numpy", "pandas", "scipy", "scikit-learn", "joblib")


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


def build_pipeline(alpha: float = _RIDGE_ALPHA) -> Pipeline:
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


# Backward-compatible alias for existing unit tests
build_sensor_pipeline = build_pipeline


# ---------------------------------------------------------------------------
# Training & artifact persistence
# ---------------------------------------------------------------------------

def train_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    alpha: float = _RIDGE_ALPHA,
    metadata: dict[str, Any] | None = None,
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
    alpha:
        Ridge regularisation parameter.
    metadata:
        Optional metadata dict written to ``model_metadata.json``.

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
    # 2. Ridge Pipeline (StandardScaler -> Ridge)
    # ------------------------------------------------------------------
    print(f"\n[TRAIN] Fitting Pipeline (StandardScaler -> Ridge, alpha={alpha}) ...")
    pipeline = build_pipeline(alpha=alpha)
    pipeline.fit(X, y)

    train_preds = pipeline.predict(X)
    train_residuals = y.values - train_preds
    train_mae = float(np.mean(np.abs(train_residuals)))
    train_rmse = float(np.sqrt(np.mean(train_residuals ** 2)))
    print(f"[TRAIN] Ridge -- Train MAE : {train_mae:.4f} mm")
    print(f"[TRAIN] Ridge -- Train RMSE: {train_rmse:.4f} mm")
    print(
        "[NOTE] Training-set error is optimistic (no held-out validation). "
        "Generalisation metrics come from out-of-fold LOTO-CV evaluation."
    )
    if metadata is not None:
        metadata = {
            **metadata,
            "final_model_training_error": {
                "description": (
                    "In-sample error of the final model on the same cuts it was fitted on. "
                    "Optimistic; not an estimate of performance on unseen tools."
                ),
                "mae_mm": round(train_mae, 6),
                "rmse_mm": round(train_rmse, 6),
                "n_samples": int(len(y)),
            },
        }

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

    # 3d. Model metadata -> model_metadata.json (if provided)
    if metadata is not None:
        metadata_path = out_dir / "model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        print(f"[SAVE] Model metadata      -> {metadata_path}")
        saved["model_metadata.json"] = metadata_path

    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python ml/train_model.py",
        description="Train verified tool wear prediction model and persist artifacts.",
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
    parser.add_argument(
        "--features",
        type=str,
        default="combined",
        choices=["combined", "sensor_only"],
        help="Feature architecture: 'combined' (77 features) or 'sensor_only' (76 features).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=_OPTIMAL_COMBINED_ALPHA,
        help=(
            "Ridge regularisation parameter alpha, used only with --no-tune  "
            f"[default: {_OPTIMAL_COMBINED_ALPHA}]"
        ),
    )
    parser.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Select alpha by grouped LOTO-CV over all tools before the final fit, and report "
            "nested LOTO-CV metrics of that procedure (default). --no-tune uses a fixed --alpha; "
            "its reported metrics are optimistic if that alpha was chosen from earlier results."
        ),
    )
    return parser.parse_args(argv)


def compute_file_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file for artifact provenance."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_package_versions() -> dict[str, str]:
    """Record installed versions of the packages that affect artifact deserialization."""
    versions: dict[str, str] = {}
    for pkg in _PROVENANCE_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def portable_path(path: Path) -> str:
    """Return ``path`` relative to the project root (POSIX style), or its file name if outside it.

    Keeps machine-specific absolute paths (and user names) out of shared metadata.
    """
    try:
        return path.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def evaluate_shipped_configuration(
    df: pd.DataFrame,
    feature_set: str,
    alpha: float,
    tuned: bool,
) -> dict[str, Any]:
    """Compute LOTO-CV metrics for exactly the configuration being shipped.

    This is the single source of truth for the out-of-fold metrics written to
    ``model_metadata.json``: nothing is copied from earlier reports or hard-coded.
    ``tuned=True`` evaluates the full tuning procedure (inner LOTO alpha selection
    inside every outer fold), matching ``--tune``.
    """
    from ml.evaluate_model import run_loto_cv

    model_key = "ridge_tuned" if tuned else "ridge"
    feature_names = get_training_feature_names(feature_set)
    results = run_loto_cv(
        df,
        alpha=alpha,
        models=[(model_key, feature_set, feature_names)],
        alpha_grid=_ALPHA_GRID,
    )
    agg = results["aggregate"][model_key]
    if agg.get("status") != "ok":
        raise RuntimeError(f"LOTO-CV of the shipped configuration failed: {agg}")

    if tuned:
        description = (
            "Nested leave-one-tool-out estimate: for each held-out tool, alpha was re-selected "
            "by inner LOTO on the remaining tools, so the held-out tool never influenced tuning. "
            "This estimates the error of the tuning procedure on tools like those in the NASA "
            "milling dataset; it is not a guarantee of performance on future tools."
        )
    else:
        description = (
            "Leave-one-tool-out estimate with a fixed alpha. Optimistic if that alpha was "
            "chosen from earlier evaluation results; not a guarantee of performance on future tools."
        )
    evaluation: dict[str, Any] = {
        "description": description,
        "source": "computed by ml/train_model.py via ml.evaluate_model.run_loto_cv at training time",
        "evaluation_strategy": results["validation_strategy"],
        "alpha_selection": "nested_inner_loto" if tuned else "fixed",
        "alpha_grid": list(_ALPHA_GRID) if tuned else [alpha],
        "mae_mm": agg["mae_mm"],
        "rmse_mm": agg["rmse_mm"],
        "r2_score": agg["r2_score"],
        "n_samples": agg["n_samples"],
        "n_folds_ok": agg["n_folds_ok"],
        "n_tools": results["n_tools"],
    }
    if tuned:
        evaluation["selected_alpha_per_fold"] = {
            fold["tool_id"]: fold["models"][model_key]["selected_alpha"]
            for fold in results["per_tool"]
            if fold.get("models", {}).get(model_key, {}).get("status") == "ok"
        }
    return evaluation


def main(argv: list[str] | None = None) -> None:
    """Main training entry point."""
    args = parse_args(argv)

    data_path: Path = args.data.resolve()
    out_dir: Path = args.out_dir.resolve()
    feature_set: str = args.features
    alpha: float = args.alpha

    # Step 1: Load and filter eligible data
    df = load_and_filter(data_path)

    # Optional: Tune alpha via inner LOTO-CV across all training tools if requested
    if args.tune:
        from ml.evaluate_model import _tune_ridge_alpha_inner_loto
        print("[TUNE] Running grouped LOTO-CV alpha selection across all tools ...")
        feat_cols = get_training_feature_names(feature_set)
        alpha = _tune_ridge_alpha_inner_loto(df, feat_cols, _ALPHA_GRID)
        print(f"[TUNE] Optimal regularisation selected: alpha = {alpha}")

    print("=" * 70)
    print("CNC Sentinel -- Model Training (Production Packaging)")
    print("=" * 70)
    print(f"  Data path   : {data_path}")
    print(f"  Output dir  : {out_dir}")
    print(f"  Target      : {TARGET_COLUMN}")
    print(f"  Feature set : {feature_set}")
    print(f"  Ridge alpha : {alpha}")
    print("=" * 70)

    # Step 2: Build feature matrix (X, y)
    X, y = build_feature_matrix(df, feature_set=feature_set)

    # Step 3: Out-of-fold evaluation of exactly this configuration (source of truth)
    print("\n[EVAL] LOTO-CV of the shipped configuration ...")
    oof_metrics = evaluate_shipped_configuration(df, feature_set, alpha, tuned=args.tune)

    # Step 4: Compute provenance metadata
    dataset_hash = compute_file_sha256(data_path) if data_path.exists() else "unknown"
    metadata = {
        "model_name": "WearRegressionRUL",
        "model_version": "1.0.0",
        "model_type": "Ridge",
        "hyperparameters": {
            "alpha": alpha,
            "fit_intercept": True,
            "random_state": 42,
        },
        "feature_set": feature_set,
        "feature_count": len(X.columns),
        "target_column": TARGET_COLUMN,
        "target_unit": "mm",
        "n_training_samples": len(X),
        "n_training_tools": int(df[_GROUP_COLUMN].nunique()),
        "training_tools": sorted(df[_GROUP_COLUMN].unique().tolist()),
        "training_dataset_path": portable_path(data_path),
        "training_dataset_sha256": dataset_hash,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": collect_package_versions(),
        "out_of_fold_metrics": oof_metrics,
    }

    # Step 5: Train on all eligible data and persist artifacts
    saved = train_and_save(X, y, out_dir, alpha=alpha, metadata=metadata)

    print("\n" + "=" * 70)
    print("Training complete. Artifacts written:")
    for name, path in saved.items():
        print(f"  {name:35s}  {path}")
    print("=" * 70)
    print(
        "\nProvenance Notes:\n"
        f"  * The final model is fitted on all {len(X)} eligible cuts with alpha={alpha}.\n"
        f"  * Out-of-fold LOTO estimate ({oof_metrics['alpha_selection']} alpha): "
        f"MAE={oof_metrics['mae_mm']:.4f} mm, RMSE={oof_metrics['rmse_mm']:.4f} mm, "
        f"R2={oof_metrics['r2_score']:.4f}\n"
        "    This is an estimate on held-out historical tools, not a guarantee for future tools.\n"
        "  * In-sample training error is recorded separately as 'final_model_training_error'."
    )


if __name__ == "__main__":
    main()
