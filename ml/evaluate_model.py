"""CNC Sentinel — Leave-One-Tool-Out Model Evaluation (P2: Model Training).

Implements Leave-One-Tool-Out Cross-Validation (LOTO-CV) over the 15 tool
cases that have at least one training-eligible cut (case_06 has no eligible
cuts and is automatically skipped).

Three models are compared on the **same** eligible held-out cuts in every fold:

  1. ``dummy``    — DummyRegressor(strategy="mean")
                   Unconditional mean of the *training* folds only.

  2. ``progress`` — StandardScaler -> Ridge on PROGRESS_BASELINE_FEATURES
                   [cutting_time_min, material_id, feed_rate_mm_rev,
                    depth_of_cut_mm]
                   Upper-bound on what a simple wear-curve model can do.

  3. ``sensor``   — StandardScaler -> Ridge on SENSOR_ONLY_FEATURES (76 cols)
                   The primary wear-regression model.

All three pipelines are fitted **inside** each training fold so that held-out
tool data never contaminates preprocessing (scaler statistics are computed
solely on the 14-tool training partition).

Eligibility policy  (per artifacts/README.md §1 and docs/ACTUAL_DATA_INSPECTION.md §6):
  * Only rows where ``training_eligible == True`` enter the CV loop.
  * Expected population: 143 cuts across 15 tool groups.
  * Case 06 contributes 0 eligible cuts and acts only as a held-out fold
    (the fold is skipped because there is nothing to predict).

Output artifact (P1-approved, Git-ignored via ``artifacts/*``):
  * ``artifacts/metrics_report.json`` — per-tool and aggregate LOTO-CV
    scores consumed by the backend ``/api/generalization`` endpoint.

Constraints:
  * Does NOT modify any existing file.
  * Does NOT install packages.
  * Does NOT claim validated RUL.
  * Does NOT use random row splits.
  * Stops with a clear error if schema, group column, or dependency is absent.

Usage:
    python ml/evaluate_model.py [--data artifacts/cut_features.csv]
                                [--out-dir artifacts]
                                [--alpha 1.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path guard: allow ``from ml.feature_config import ...`` regardless of cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Dependency guard — fail early with an actionable message
# ---------------------------------------------------------------------------
def _check_dependencies() -> None:
    missing = []
    for pkg in ("numpy", "pandas", "sklearn", "joblib"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Required package(s) not found: {', '.join(missing)}\n"
            "Install ML dependencies first:\n"
            "    pip install -r ml/requirements.txt"
        )


_check_dependencies()

# ---------------------------------------------------------------------------
# Imports (after dependency guard)
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ml.feature_config import (  # noqa: E402
    COMBINED_FEATURES,
    IDENTIFIER_COLUMNS,
    PROGRESS_BASELINE_FEATURES,
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

# Group column that defines tool identity for LOTO splitting
_GROUP_COLUMN = "tool_id"

# Certified eligible population (per artifacts/README.md §1)
_EXPECTED_ELIGIBLE_COUNT = 143

# Default Ridge regularisation (matches train_model.py default)
_RIDGE_ALPHA = 1.0
_DEFAULT_ALPHA_GRID: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]

# All columns the evaluation script must find in the CSV
_REQUIRED_COLUMNS: list[str] = (
    IDENTIFIER_COLUMNS
    + QUALITY_COLUMNS
    + [TARGET_COLUMN, _GROUP_COLUMN]
    + SENSOR_ONLY_FEATURES
    + PROGRESS_BASELINE_FEATURES
)

# Model specifications: (key, feature_set_name, feature_list)
_MODEL_SPECS: list[tuple[str, str, list[str]]] = [
    ("dummy", "dummy", []),
    ("progress", "progress_baseline", list(PROGRESS_BASELINE_FEATURES)),
    ("sensor", "sensor_only", list(SENSOR_ONLY_FEATURES)),
]

# Extended candidate specifications for controlled multi-model comparisons
_EXTENDED_MODEL_SPECS: list[tuple[str, str, list[str]]] = [
    ("dummy", "dummy", []),
    ("progress", "progress_baseline", list(PROGRESS_BASELINE_FEATURES)),
    ("sensor", "sensor_only", list(SENSOR_ONLY_FEATURES)),
    ("combined", "combined", list(COMBINED_FEATURES)),
    ("combined_tuned", "combined", list(COMBINED_FEATURES)),
    ("extra_trees", "combined", list(COMBINED_FEATURES)),
]

# Generic keys used by train_model.py to evaluate the exact shipped configuration:
# "ridge" = fixed alpha, "ridge_tuned" = alpha selected by inner LOTO in every outer fold.
_TUNED_MODEL_KEYS: tuple[str, ...] = ("combined_tuned", "ridge_tuned")
_RIDGE_MODEL_KEYS: tuple[str, ...] = ("progress", "sensor", "combined", "ridge") + _TUNED_MODEL_KEYS


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------

def load_and_filter(data_path: Path) -> pd.DataFrame:
    """Load the feature table, validate schema, and apply the eligibility filter.

    Returns
    -------
    pd.DataFrame
        Rows where ``training_eligible == True``, with the group column intact.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If required columns are missing or the eligible population is empty.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {data_path}\n"
            "Run the feature-extraction pipeline first:\n"
            "    python ml/prepare_data.py"
        )

    df = pd.read_csv(data_path)

    # Schema check -- surface missing columns as a blocker
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing {len(missing_cols)} required column(s): "
            f"{missing_cols}\nFound columns: {list(df.columns)}"
        )

    # Group column must be non-null
    if df[_GROUP_COLUMN].isna().any():
        raise ValueError(
            f"Group column '{_GROUP_COLUMN}' contains NaN values. "
            "Cannot perform LOTO-CV without complete tool identifiers."
        )

    # Normalise boolean columns from CSV string variants
    for bool_col in ("training_eligible", "is_valid_run"):
        df[bool_col] = df[bool_col].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes")
        )

    train_df = df[df["training_eligible"]].copy().reset_index(drop=True)

    if train_df.empty:
        raise ValueError(
            "No training-eligible rows found (training_eligible == True). "
            "Check that cut_features.csv was produced by the full pipeline."
        )

    actual = len(train_df)
    if actual != _EXPECTED_ELIGIBLE_COUNT:
        print(
            f"[WARNING] Expected {_EXPECTED_ELIGIBLE_COUNT} eligible rows, "
            f"found {actual}. Proceeding — verify the feature table is current.",
            file=sys.stderr,
        )
    else:
        print(f"[INFO] Eligible population: {actual} cuts (matches certified count).")

    # Target must be non-null within eligible subset
    null_target = train_df[TARGET_COLUMN].isna().sum()
    if null_target > 0:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' has {null_target} NaN(s) in eligible subset. "
            "The eligibility filter should guarantee non-null labels."
        )

    return train_df


# ---------------------------------------------------------------------------
# Pipeline factories (re-created fresh for every fold)
# ---------------------------------------------------------------------------

def _make_pipeline(model_key: str, alpha: float) -> Pipeline | DummyRegressor | ExtraTreesRegressor:
    """Return a fresh, unfitted estimator for the given model key."""
    if model_key == "dummy":
        return DummyRegressor(strategy="mean")
    if model_key in _RIDGE_MODEL_KEYS:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, fit_intercept=True, random_state=42)),
            ]
        )
    if model_key == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
        )
    raise ValueError(f"Unknown model_key: {model_key}")


def _tune_ridge_alpha_inner_loto(
    train_df: pd.DataFrame,
    feat_cols: list[str],
    alpha_grid: list[float],
) -> float:
    """Select best Ridge alpha using inner Leave-One-Tool-Out CV on training data only.

    Strictly guarantees that no data or labels from the outer held-out tool are seen during tuning.
    """
    inner_tool_ids = sorted(train_df[_GROUP_COLUMN].unique())
    if len(inner_tool_ids) <= 1:
        return 1.0

    best_alpha = 1.0
    best_inner_mae = float("inf")

    for candidate_alpha in alpha_grid:
        inner_abs_residuals: list[float] = []
        for inner_held_out in inner_tool_ids:
            inner_test_mask = train_df[_GROUP_COLUMN] == inner_held_out
            inner_train_mask = ~inner_test_mask

            in_train_df = train_df[inner_train_mask]
            in_test_df = train_df[inner_test_mask]

            if in_train_df.empty or in_test_df.empty:
                continue

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=candidate_alpha, fit_intercept=True, random_state=42)),
            ])
            X_in_train = in_train_df[feat_cols].astype(float)
            y_in_train = in_train_df[TARGET_COLUMN].astype(float)
            X_in_test = in_test_df[feat_cols].astype(float)
            y_in_test = in_test_df[TARGET_COLUMN].astype(float).values

            pipe.fit(X_in_train, y_in_train)
            y_in_pred = pipe.predict(X_in_test)
            inner_abs_residuals.extend(np.abs(y_in_test - y_in_pred))

        if inner_abs_residuals:
            mean_inner_mae = float(np.mean(inner_abs_residuals))
            if mean_inner_mae < best_inner_mae:
                best_inner_mae = mean_inner_mae
                best_alpha = candidate_alpha

    return best_alpha


# ---------------------------------------------------------------------------
# Leave-One-Tool-Out cross-validation
# ---------------------------------------------------------------------------

def run_loto_cv(
    df: pd.DataFrame,
    alpha: float = _RIDGE_ALPHA,
    models: list[tuple[str, str, list[str]]] | None = None,
    alpha_grid: list[float] | None = None,
) -> dict[str, Any]:
    """Execute Leave-One-Tool-Out CV and return a structured results dict.

    For each fold:
      - The held-out tool's eligible cuts form the test set.
      - All other tools' eligible cuts form the training set.
      - Estimators are fitted on the training set, with preprocessing
        (StandardScaler) fitted **only** on training rows in that fold.
      - For nested hyperparameter selection (e.g. combined_tuned), inner CV is
        run solely on the training fold to select alpha.
      - Predictions are made on the held-out tool's rows.
      - MAE and RMSE are computed in mm, and pooled R² is evaluated.

    Parameters
    ----------
    df:
        Training-eligible rows (output of ``load_and_filter``).
    alpha:
        Ridge regularisation strength for fixed-alpha models.
    models:
        Optional list of (model_key, feature_set_name, feature_list).
        Defaults to ``_MODEL_SPECS`` (dummy, progress, sensor).
    alpha_grid:
        List of candidate alphas for inner CV tuning (used by combined_tuned).

    Returns
    -------
    dict with keys ``per_tool`` (list of per-fold dicts) and ``aggregate``
    (dict of model -> {mae, rmse, r2_score, n_samples}).
    """
    model_specs = models if models is not None else _MODEL_SPECS
    grid = alpha_grid if alpha_grid is not None else _DEFAULT_ALPHA_GRID

    tool_ids = sorted(df[_GROUP_COLUMN].unique())
    n_tools = len(tool_ids)
    print(f"\n[LOTO-CV] {n_tools} tool folds: {tool_ids}")

    model_keys = [spec[0] for spec in model_specs]
    feature_sets = {spec[0]: spec[2] for spec in model_specs}

    # Accumulators: model_key -> list of residuals array
    all_residuals: dict[str, list[np.ndarray]] = {k: [] for k in model_keys}
    per_tool_results: list[dict[str, Any]] = []

    for fold_idx, held_out_tool in enumerate(tool_ids, start=1):
        test_mask = df[_GROUP_COLUMN] == held_out_tool
        train_mask = ~test_mask

        test_df = df[test_mask]
        train_df = df[train_mask]

        n_test = len(test_df)
        n_train = len(train_df)

        print(
            f"\n[Fold {fold_idx:2d}/{n_tools}]  held-out={held_out_tool}  "
            f"train={n_train} cuts  test={n_test} cuts"
        )

        fold_result: dict[str, Any] = {
            "tool_id": held_out_tool,
            "fold": fold_idx,
            "n_train": n_train,
            "n_test": n_test,
            "models": {},
            "status": "ok",
        }

        if n_train == 0:
            print(
                f"  [SKIP] Training set empty for fold {fold_idx} "
                f"(held-out={held_out_tool}). Fold recorded as failed."
            )
            fold_result["status"] = "failed_empty_train"
            per_tool_results.append(fold_result)
            continue

        if n_test == 0:
            print(
                f"  [SKIP] Test set empty (0 eligible cuts for {held_out_tool}). "
                "Fold skipped — no predictions possible."
            )
            fold_result["status"] = "skipped_empty_test"
            per_tool_results.append(fold_result)
            continue

        y_test = test_df[TARGET_COLUMN].astype(float).values

        for model_key in model_keys:
            feat_cols = feature_sets[model_key]

            if model_key == "dummy":
                X_train = train_df[[TARGET_COLUMN]].astype(float)
                X_test = test_df[[TARGET_COLUMN]].astype(float)
            else:
                X_train = train_df[feat_cols].astype(float)
                X_test = test_df[feat_cols].astype(float)

            y_train = train_df[TARGET_COLUMN].astype(float)

            # Fresh estimator — preprocessing and tuning fitted on training fold only
            if model_key in _TUNED_MODEL_KEYS:
                selected_alpha = _tune_ridge_alpha_inner_loto(train_df, feat_cols, grid)
                estimator = _make_pipeline(model_key, selected_alpha)
            else:
                selected_alpha = alpha
                estimator = _make_pipeline(model_key, alpha)

            try:
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] {model_key}: {exc}")
                fold_result["models"][model_key] = {
                    "status": "error",
                    "error": str(exc),
                }
                continue

            residuals = y_test - y_pred
            mae = float(np.mean(np.abs(residuals)))
            rmse = float(np.sqrt(np.mean(residuals ** 2)))

            model_metrics: dict[str, Any] = {
                "mae_mm": round(mae, 6),
                "rmse_mm": round(rmse, 6),
                "n_test": n_test,
                "status": "ok",
            }
            if model_key in _TUNED_MODEL_KEYS:
                model_metrics["selected_alpha"] = selected_alpha

            fold_result["models"][model_key] = model_metrics
            all_residuals[model_key].append(residuals)

            alpha_info = f" (alpha={selected_alpha})" if model_key in _TUNED_MODEL_KEYS else ""
            print(f"  {model_key:<16}  MAE={mae:.4f} mm  RMSE={rmse:.4f} mm{alpha_info}")

        per_tool_results.append(fold_result)

    # Aggregate metrics across all successful folds
    print("\n[AGGREGATE]")
    y_true_all = df[TARGET_COLUMN].astype(float).values
    ss_tot = float(np.sum((y_true_all - np.mean(y_true_all)) ** 2))

    aggregate: dict[str, dict[str, Any]] = {}
    for model_key in model_keys:
        stacked = all_residuals[model_key]
        if not stacked:
            aggregate[model_key] = {"status": "no_successful_folds"}
            print(f"  {model_key:<16}  no successful folds")
            continue

        all_res = np.concatenate(stacked)
        n_total = len(all_res)
        agg_mae = float(np.mean(np.abs(all_res)))
        agg_rmse = float(np.sqrt(np.mean(all_res ** 2)))
        ss_res = float(np.sum(all_res ** 2))
        agg_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        n_folds_ok = len(stacked)

        aggregate[model_key] = {
            "mae_mm": round(agg_mae, 6),
            "rmse_mm": round(agg_rmse, 6),
            "r2_score": round(agg_r2, 6),
            "n_samples": n_total,
            "n_folds_ok": n_folds_ok,
            "status": "ok",
        }
        print(
            f"  {model_key:<16}  MAE={agg_mae:.4f} mm  RMSE={agg_rmse:.4f} mm  "
            f"R2={agg_r2:.4f}  n={n_total}  folds_ok={n_folds_ok}/{n_tools}"
        )

    # Count failed / skipped folds
    failed_folds = [
        r["tool_id"]
        for r in per_tool_results
        if r["status"] not in ("ok",)
    ]

    return {
        "validation_strategy": "leave_one_tool_out",
        "group_column": _GROUP_COLUMN,
        "target_column": TARGET_COLUMN,
        "n_tools": n_tools,
        "n_eligible_cuts": len(df),
        "ridge_alpha": alpha,
        "models_compared": model_keys,
        "feature_counts": {
            spec[0]: len(spec[2]) for spec in model_specs
        },
        "failed_or_skipped_folds": failed_folds,
        "per_tool": per_tool_results,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_metrics_report(results: dict[str, Any], out_dir: Path) -> Path:
    """Write baseline LOTO-CV results to artifacts/metrics_report.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "metrics_report.json"

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n[SAVE] Metrics report -> {report_path}")
    return report_path


def save_comparison_report(results: dict[str, Any], out_dir: Path) -> Path:
    """Write multi-model comparison results to artifacts/model_comparison_report.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "model_comparison_report.json"

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n[SAVE] Model comparison report -> {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python ml/evaluate_model.py",
        description=(
            "Leave-One-Tool-Out CV: DummyRegressor vs. progress baseline "
            "vs. sensor-only Ridge vs. combined models."
        ),
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
            f"Output directory for report artifacts  "
            f"[default: {_DEFAULT_OUT_DIR.relative_to(_PROJECT_ROOT)}]"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=_RIDGE_ALPHA,
        metavar="ALPHA",
        help=f"Ridge regularisation strength  [default: {_RIDGE_ALPHA}]",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comprehensive model comparison across all 6 candidate configurations.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main evaluation entry point."""
    args = parse_args(argv)

    data_path: Path = args.data.resolve()
    out_dir: Path = args.out_dir.resolve()
    alpha: float = args.alpha
    is_compare: bool = args.compare

    models_to_run = _EXTENDED_MODEL_SPECS if is_compare else _MODEL_SPECS

    print("=" * 70)
    print("CNC Sentinel -- Leave-One-Tool-Out Evaluation (P2)")
    print("=" * 70)
    print(f"  Data path    : {data_path}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Target       : {TARGET_COLUMN}")
    print(f"  Group column : {_GROUP_COLUMN}")
    print(f"  Ridge alpha  : {alpha}")
    print(f"  Comparison   : {'Full Candidate Comparison' if is_compare else 'Baseline Models'}")
    print(f"  Models ({len(models_to_run)}) : {', '.join(m[0] for m in models_to_run)}")
    print("=" * 70)

    # Load & validate
    df = load_and_filter(data_path)

    # Run LOTO-CV
    results = run_loto_cv(df, alpha=alpha, models=models_to_run)

    # Persist artifact
    if is_compare:
        report_path = save_comparison_report(results, out_dir)
    else:
        report_path = save_metrics_report(results, out_dir)

    # Final summary table
    print("\n" + "=" * 78)
    print("Evaluation complete — aggregate LOTO-CV results:")
    print(f"  {'Model':<16}  {'MAE (mm)':>10}  {'RMSE (mm)':>10}  {'Pooled R²':>10}  {'n_samples':>10}  {'folds_ok':>10}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for model_key, agg in results["aggregate"].items():
        if agg.get("status") != "ok":
            print(f"  {model_key:<16}  {'--':>10}  {'--':>10}  {'--':>10}  {'--':>10}  {'--':>10}")
            continue
        print(
            f"  {model_key:<16}  "
            f"{agg['mae_mm']:>10.4f}  "
            f"{agg['rmse_mm']:>10.4f}  "
            f"{agg.get('r2_score', 0.0):>10.4f}  "
            f"{agg['n_samples']:>10}  "
            f"{agg['n_folds_ok']:>10}"
        )

    if results["failed_or_skipped_folds"]:
        print(
            f"\n  [NOTE] {len(results['failed_or_skipped_folds'])} fold(s) "
            f"failed or skipped: {results['failed_or_skipped_folds']}"
        )

    print(f"\n  Artifact: {report_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
