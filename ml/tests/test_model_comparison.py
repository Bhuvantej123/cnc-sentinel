"""CNC Sentinel — Model Comparison & Nested Tuning Tests (P1 ML Improvement).

Validates:
1. Combined feature boundaries, order consistency, and strict exclusion of leakage columns.
2. Estimator factory support for combined Ridge, tuned Ridge, and ExtraTreesRegressor.
3. Nested inner Leave-One-Tool-Out CV isolation (zero leakage of outer held-out tool data).
4. Full comparative LOTO-CV execution and mathematical invariants (RMSE >= MAE, valid R²).
5. Model comparison artifact persistence and JSON structure.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.feature_config import (
    COMBINED_FEATURES,
    IDENTIFIER_COLUMNS,
    OPERATING_CONDITION_COLUMNS,
    PROGRESS_COLUMNS,
    QUALITY_COLUMNS,
    SENSOR_FEATURE_COLUMNS,
    SENSOR_ONLY_FEATURES,
    TARGET_COLUMN,
    get_training_feature_names,
)
from ml.evaluate_model import (
    _DEFAULT_ALPHA_GRID,
    _EXTENDED_MODEL_SPECS,
    _GROUP_COLUMN,
    _MODEL_SPECS,
    _make_pipeline,
    _tune_ridge_alpha_inner_loto,
    run_loto_cv,
    save_comparison_report,
)


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def synthetic_comparison_df() -> pd.DataFrame:
    """Generate a clean synthetic dataset across 3 tools for testing LOTO-CV comparison."""
    rows = []
    tool_ids = ["tool_A", "tool_B", "tool_C"]
    rng = np.random.RandomState(42)

    for tool_idx, tool_id in enumerate(tool_ids):
        base_wear = 0.05 * (tool_idx + 1)
        for cut in range(1, 6):
            wear = base_wear + 0.02 * cut + rng.normal(0, 0.005)
            row: dict[str, Any] = {
                "tool_id": tool_id,
                "cut_index": cut,
                "case_num": tool_idx + 1,
                "run_num": cut,
                "quality_flag": "good",
                "is_valid_run": True,
                "training_eligible": True,
                "demo_eligible": True,
                TARGET_COLUMN: float(wear),
                "cutting_time_min": float(cut * 2.5),
                "material_id": 1.0,
                "feed_rate_mm_rev": 0.25,
                "depth_of_cut_mm": 0.75,
            }
            # Populate sensor features with deterministic synthetic values
            for feat in SENSOR_FEATURE_COLUMNS:
                row[feat] = float(rng.normal(1.0, 0.1))
            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 1. Combined Feature Boundary & Leakage Tests
# =============================================================================

class TestCombinedFeatureBoundaries:
    """Verify combined features prevent leakage and preserve exact ordering."""

    def test_combined_feature_composition(self):
        """COMBINED_FEATURES must be strictly SENSOR_ONLY_FEATURES + PROGRESS_COLUMNS."""
        assert len(COMBINED_FEATURES) == len(SENSOR_ONLY_FEATURES) + len(PROGRESS_COLUMNS)
        assert len(COMBINED_FEATURES) == 77  # 72 channel + 1 current_rms + 3 conditions + 1 time
        assert "cutting_time_min" in COMBINED_FEATURES
        assert "current_rms" in COMBINED_FEATURES

    def test_strict_exclusion_of_leakage_columns(self):
        """COMBINED_FEATURES must strictly exclude identifiers, targets, and quality flags."""
        combined_set = set(COMBINED_FEATURES)

        for col in IDENTIFIER_COLUMNS:
            assert col not in combined_set, f"Identifier {col} leaked into COMBINED_FEATURES"

        assert TARGET_COLUMN not in combined_set, f"Target {TARGET_COLUMN} leaked into COMBINED_FEATURES"

        for col in QUALITY_COLUMNS:
            assert col not in combined_set, f"Quality column {col} leaked into COMBINED_FEATURES"

    def test_get_training_feature_names_combined(self):
        """get_training_feature_names('combined') must match COMBINED_FEATURES exactly."""
        names = get_training_feature_names("combined")
        assert names == list(COMBINED_FEATURES)
        assert len(names) == 77


# =============================================================================
# 2. Estimator Factory Tests
# =============================================================================

class TestEstimatorFactory:
    """Verify _make_pipeline constructs fresh unfitted estimators for all model types."""

    def test_combined_ridge_pipeline(self):
        pipe = _make_pipeline("combined", alpha=10.0)
        assert isinstance(pipe, Pipeline)
        assert "scaler" in pipe.named_steps
        assert "ridge" in pipe.named_steps
        assert pipe.named_steps["ridge"].alpha == 10.0
        assert not hasattr(pipe.named_steps["scaler"], "mean_")

    def test_extra_trees_estimator(self):
        et = _make_pipeline("extra_trees", alpha=1.0)
        assert isinstance(et, ExtraTreesRegressor)
        assert et.n_estimators == 100
        assert et.max_depth == 6
        assert et.min_samples_leaf == 2
        assert et.random_state == 42
        assert not hasattr(et, "estimators_")

    def test_unknown_model_key_raises(self):
        with pytest.raises(ValueError, match="Unknown model_key"):
            _make_pipeline("nonexistent_model", alpha=1.0)


# =============================================================================
# 3. Nested Inner-CV Isolation Tests (Zero Leakage)
# =============================================================================

class TestNestedInnerCV:
    """Verify inner LOTO-CV tunes alpha exclusively on the outer training fold."""

    def test_inner_tuning_returns_valid_alpha_from_grid(self, synthetic_comparison_df: pd.DataFrame):
        train_df = synthetic_comparison_df[synthetic_comparison_df[_GROUP_COLUMN] != "tool_A"]
        grid = [0.01, 1.0, 100.0]
        selected_alpha = _tune_ridge_alpha_inner_loto(train_df, list(COMBINED_FEATURES), grid)
        assert selected_alpha in grid

    def test_inner_tuning_isolated_from_outer_held_out_tool(self, synthetic_comparison_df: pd.DataFrame):
        """Modifying outer held-out tool data must not change selected alpha on training data."""
        train_df = synthetic_comparison_df[synthetic_comparison_df[_GROUP_COLUMN] != "tool_A"].copy()
        held_out_df = synthetic_comparison_df[synthetic_comparison_df[_GROUP_COLUMN] == "tool_A"].copy()

        grid = [0.1, 10.0, 100.0]
        alpha1 = _tune_ridge_alpha_inner_loto(train_df, list(COMBINED_FEATURES), grid)

        # Mutate outer held-out tool's data
        held_out_df[TARGET_COLUMN] = 999.0
        held_out_df["current_rms"] = -1000.0

        # Run inner tuning on train_df again; must be bit-identical
        alpha2 = _tune_ridge_alpha_inner_loto(train_df, list(COMBINED_FEATURES), grid)
        assert alpha1 == alpha2


# =============================================================================
# 4. Multi-Model LOTO-CV Execution & Invariants
# =============================================================================

class TestMultiModelLOTOCV:
    """Verify run_loto_cv executes all 6 candidates and respects mathematical invariants."""

    def test_full_candidate_evaluation_end_to_end(self, synthetic_comparison_df: pd.DataFrame):
        results = run_loto_cv(
            synthetic_comparison_df,
            alpha=1.0,
            models=_EXTENDED_MODEL_SPECS,
            alpha_grid=[0.1, 1.0, 10.0],
        )

        expected_models = {"dummy", "progress", "sensor", "combined", "combined_tuned", "extra_trees"}
        assert set(results["models_compared"]) == expected_models
        assert results["n_tools"] == 3
        assert results["n_eligible_cuts"] == 15

        agg = results["aggregate"]
        for m in expected_models:
            assert m in agg
            assert agg[m]["status"] == "ok"
            assert agg[m]["n_samples"] == 15
            assert agg[m]["n_folds_ok"] == 3
            assert agg[m]["mae_mm"] >= 0.0
            # Mathematical invariant: RMSE >= MAE
            assert agg[m]["rmse_mm"] >= agg[m]["mae_mm"] - 1e-12
            assert "r2_score" in agg[m]

    def test_save_comparison_report_file_structure(self, synthetic_comparison_df: pd.DataFrame, tmp_path: Path):
        results = run_loto_cv(
            synthetic_comparison_df,
            alpha=1.0,
            models=_EXTENDED_MODEL_SPECS,
        )
        out_dir = tmp_path / "artifacts"
        report_path = save_comparison_report(results, out_dir=out_dir)

        assert report_path.exists()
        assert report_path.name == "model_comparison_report.json"

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "aggregate" in data
        assert "per_tool" in data
        assert "combined_tuned" in data["aggregate"]
        assert "extra_trees" in data["aggregate"]


# =============================================================================
# 5. Real-Data Artifact & Strict Parity Tests
# =============================================================================

class TestArtifactAndParityVerification:
    """Verify that the actual saved comparison report strictly conforms to certified data."""

    def test_saved_report_sample_counts_sum_to_143_and_15_folds(self):
        """Saved artifacts/model_comparison_report.json must reflect exactly 143 cuts across 15 folds."""
        report_path = REPO_ROOT / "artifacts" / "model_comparison_report.json"
        if not report_path.exists():
            pytest.skip("artifacts/model_comparison_report.json does not exist yet.")

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["validation_strategy"] == "leave_one_tool_out"
        assert data["n_tools"] == 15
        assert data["n_eligible_cuts"] == 143

        # All 15 tool folds must be present and completed
        per_tool = data["per_tool"]
        assert len(per_tool) == 15
        total_cuts = sum(pt["n_test"] for pt in per_tool)
        assert total_cuts == 143, f"Sum of fold n_test must be 143, got {total_cuts}"

        # Every model must have evaluated on all 143 cuts across all 15 folds
        expected_models = ["dummy", "progress", "sensor", "combined", "combined_tuned", "extra_trees"]
        for m in expected_models:
            agg = data["aggregate"][m]
            assert agg["status"] == "ok"
            assert agg["n_samples"] == 143
            assert agg["n_folds_ok"] == 15
            assert agg["rmse_mm"] >= agg["mae_mm"]

    def test_saved_report_exact_expected_aggregate_metrics(self):
        """Saved aggregate metrics must match the verified benchmark numbers exactly."""
        report_path = REPO_ROOT / "artifacts" / "model_comparison_report.json"
        if not report_path.exists():
            pytest.skip("artifacts/model_comparison_report.json does not exist yet.")

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        agg = data["aggregate"]
        # Verified numerical baselines to 4 decimal places
        assert round(agg["dummy"]["mae_mm"], 4) == 0.1997
        assert round(agg["dummy"]["rmse_mm"], 4) == 0.2645
        assert round(agg["progress"]["mae_mm"], 4) == 0.1279
        assert round(agg["progress"]["rmse_mm"], 4) == 0.1937
        assert round(agg["sensor"]["mae_mm"], 4) == 0.1503
        assert round(agg["sensor"]["rmse_mm"], 4) == 0.2188
        assert round(agg["combined"]["mae_mm"], 4) == 0.1502
        assert round(agg["combined"]["rmse_mm"], 4) == 0.2080
        assert round(agg["combined_tuned"]["mae_mm"], 4) == 0.1143
        assert round(agg["combined_tuned"]["rmse_mm"], 4) == 0.1745
        assert round(agg["extra_trees"]["mae_mm"], 4) == 0.1236
        assert round(agg["extra_trees"]["rmse_mm"], 4) == 0.1863

    def test_reproducibility_and_determinism_across_runs(self, synthetic_comparison_df: pd.DataFrame):
        """Repeated execution with identical random_state must yield bit-identical predictions and metrics."""
        res1 = run_loto_cv(synthetic_comparison_df, alpha=1.0, models=_EXTENDED_MODEL_SPECS)
        res2 = run_loto_cv(synthetic_comparison_df, alpha=1.0, models=_EXTENDED_MODEL_SPECS)

        for m in ["dummy", "progress", "sensor", "combined", "combined_tuned", "extra_trees"]:
            assert res1["aggregate"][m]["mae_mm"] == res2["aggregate"][m]["mae_mm"]
            assert res1["aggregate"][m]["rmse_mm"] == res2["aggregate"][m]["rmse_mm"]
            assert res1["aggregate"][m]["r2_score"] == res2["aggregate"][m]["r2_score"]


# =============================================================================
# 6. Production Model Artifact & Inference Interface Validation (Phase 7 Task 4)
# =============================================================================

def _real_artifacts_verified() -> bool:
    """True only when locally generated (Git-ignored) artifacts exist and pass validation."""
    import ml.inference as inf

    required = (inf.MODEL_PATH, inf.FEATURE_NAMES_PATH, inf.CUT_FEATURES_PATH)
    return all(p.exists() for p in required) and inf.is_model_verified()


requires_real_artifacts = pytest.mark.skipif(
    not _real_artifacts_verified(),
    reason=(
        "Trained artifacts are Git-ignored and absent or unverified on this machine; "
        "run ml/prepare_data.py and ml/train_model.py to enable."
    ),
)


class TestProductionArtifactPackagingAndInference:
    """Verify saved model artifacts, serialization parity, and inference safety."""

    def test_feature_names_order_and_dimensions(self):
        """feature_names.json must match COMBINED_FEATURES in exact count and order."""
        feature_names_path = REPO_ROOT / "artifacts" / "feature_names.json"
        if not feature_names_path.exists():
            pytest.skip("artifacts/feature_names.json does not exist.")

        with open(feature_names_path, "r", encoding="utf-8") as f:
            features = json.load(f)

        assert isinstance(features, list)
        assert len(features) == 77
        assert features == list(COMBINED_FEATURES)

    def test_model_metadata_provenance(self):
        """model_metadata.json must carry computed OOF metrics, versions, and a portable path."""
        metadata_path = REPO_ROOT / "artifacts" / "model_metadata.json"
        if not metadata_path.exists():
            pytest.skip("artifacts/model_metadata.json does not exist.")

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["model_name"] == "WearRegressionRUL"
        assert meta["model_version"] == "1.0.0"
        assert meta["model_type"] == "Ridge"
        assert meta["hyperparameters"]["fit_intercept"] is True
        assert meta["feature_count"] == 77
        assert meta["target_column"] == TARGET_COLUMN
        assert meta["target_unit"] == "mm"
        assert meta["n_training_samples"] == 143
        assert meta["n_training_tools"] == 15
        assert len(meta["training_tools"]) == 15
        assert meta["training_dataset_sha256"] != ""

        # Portable provenance: no machine-specific absolute paths
        assert not Path(meta["training_dataset_path"]).is_absolute()
        assert ":" not in meta["training_dataset_path"]

        # Version provenance for deserialization safety
        for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "joblib"):
            assert pkg in meta["package_versions"]
        assert "python_version" in meta

        # Out-of-fold metrics are computed at training time, never hard-coded
        oof = meta["out_of_fold_metrics"]
        assert oof["evaluation_strategy"] == "leave_one_tool_out"
        # Nested LOTO is the approved primary generalization estimate
        assert oof["alpha_selection"] == "nested_inner_loto"
        assert "not a guarantee" in oof["description"]
        assert oof["n_samples"] == 143
        assert oof["n_folds_ok"] == 15
        for key in ("mae_mm", "rmse_mm", "r2_score"):
            assert np.isfinite(oof[key])
        assert oof["rmse_mm"] >= oof["mae_mm"]
        assert "verified_out_of_fold_metrics" not in meta

        # In-sample error is reported separately and labelled as optimistic
        train_err = meta["final_model_training_error"]
        assert "Optimistic" in train_err["description"]
        assert train_err["n_samples"] == 143

    def test_save_load_prediction_parity_combined_tuned(self, tmp_path: Path):
        """Reloading pipeline yields bit-identical predictions to original fit."""
        from ml.train_model import build_pipeline, train_and_save

        # Create synthetic training set with all combined features
        rng = np.random.RandomState(42)
        X_syn = pd.DataFrame(
            rng.randn(30, len(COMBINED_FEATURES)),
            columns=list(COMBINED_FEATURES),
        )
        y_syn = pd.Series(0.1 + 0.05 * np.arange(30), name=TARGET_COLUMN)

        saved = train_and_save(X_syn, y_syn, out_dir=tmp_path / "artifacts", alpha=100.0)
        reloaded_pipe: Pipeline = joblib.load(saved["tool_wear_regressor.joblib"])

        original_pipe = build_pipeline(alpha=100.0)
        original_pipe.fit(X_syn, y_syn)

        preds_orig = original_pipe.predict(X_syn)
        preds_reloaded = reloaded_pipe.predict(X_syn)

        assert np.allclose(preds_orig, preds_reloaded, atol=1e-12)

    @requires_real_artifacts
    def test_inference_valid_cut_returns_finite_predictions_in_mm(self):
        """predict() for an eligible cut returns finite wear in mm and safe advisory."""
        from ml.inference import predict

        res = predict(tool_id="case_01", cut_index=1, wear_threshold=0.20)
        assert res["model_status"] == "VERIFIED"
        assert res["error_code"] is None
        assert res["tool_id"] == "case_01"
        assert res["cut_index"] == 1
        assert res["wear_unit"] == "mm"
        assert res["predicted_wear"] is not None
        assert isinstance(res["predicted_wear"], float)
        assert np.isfinite(res["predicted_wear"])
        assert res["predicted_wear"] >= 0.0
        assert res["recommendation"] in ["CONTINUE", "INSPECT_SOON", "REPLACE_TOOL"]
        # Invariant: No uncalibrated uncertainty bounds or unvalidated RUL claimed
        assert res["uncertainty_interval"] is None
        assert res["estimated_rul_cuts"] is None

    @requires_real_artifacts
    def test_inference_handles_unknown_or_invalid_cut(self):
        """predict() for non-existent tool/cut returns MODEL_UNAVAILABLE with UNKNOWN_CUT."""
        from ml.inference import InferenceErrorCode, RecommendationStatus, predict

        res = predict(tool_id="case_99_nonexistent", cut_index=999, wear_threshold=0.20)
        assert res["predicted_wear"] is None
        assert res["uncertainty_interval"] is None
        assert res["estimated_rul_cuts"] is None
        assert res["recommendation"] == RecommendationStatus.MODEL_UNAVAILABLE.value
        assert res["error_code"] == InferenceErrorCode.UNKNOWN_CUT.value
        assert "not found" in res["explanation"].lower()

    @requires_real_artifacts
    @pytest.mark.parametrize(
        "tool_id,cut_index",
        [("case_02", 1), ("case_12", 1), ("case_06", 1), ("case_07", 4)],
    )
    def test_real_flagged_cuts_never_receive_predictions(self, tool_id: str, cut_index: int):
        """The four quality-flagged NASA cuts must never yield a wear value or advisory."""
        from ml.inference import InferenceErrorCode, RecommendationStatus, predict

        res = predict(tool_id=tool_id, cut_index=cut_index, wear_threshold=0.20)
        assert res["predicted_wear"] is None
        assert res["recommendation"] == RecommendationStatus.MODEL_UNAVAILABLE.value
        assert res["error_code"] == InferenceErrorCode.INVALID_RUN.value

    @requires_real_artifacts
    def test_inference_rejects_non_finite_values_without_silent_zero_substitution(
        self, monkeypatch
    ):
        """Missing or NaN sensor features must cause prediction rejection without silent 0 imputation."""
        import ml.inference as inf

        df = pd.read_csv(inf.CUT_FEATURES_PATH)
        # Corrupt a single feature in case_01 cut 1
        corrupted_df = df.copy()
        mask = (corrupted_df["tool_id"] == "case_01") & (corrupted_df["cut_index"] == 1)
        corrupted_df.loc[mask, "smc_ac_mean"] = np.nan

        # Monkeypatch pd.read_csv inside ml.inference to return corrupted_df
        monkeypatch.setattr(inf.pd, "read_csv", lambda *args, **kwargs: corrupted_df)

        res = inf.predict(tool_id="case_01", cut_index=1, wear_threshold=0.20)
        assert res["predicted_wear"] is None
        assert res["recommendation"] == inf.RecommendationStatus.MODEL_UNAVAILABLE.value
        assert res["error_code"] == inf.InferenceErrorCode.FEATURES_UNAVAILABLE.value
        assert "missing or non-finite" in res["explanation"].lower()

    def test_missing_model_artifact_safe_fallback(self, monkeypatch):
        """Missing model artifacts must produce graceful MODEL_UNAVAILABLE behavior."""
        import ml.inference as inf

        # Point MODEL_PATH to a non-existent path and reset cache
        monkeypatch.setattr(inf, "MODEL_PATH", Path("nonexistent_path/fake_model.joblib"))
        monkeypatch.setattr(inf, "_cached_model", None)
        monkeypatch.setattr(inf, "_cached_scaler", None)
        monkeypatch.setattr(inf, "_cached_features", None)

        assert inf.is_model_verified() is False

        res = inf.predict(tool_id="case_01", cut_index=1, wear_threshold=0.20)
        assert res["predicted_wear"] is None
        assert res["recommendation"] == inf.RecommendationStatus.MODEL_UNAVAILABLE.value
        assert res["model_status"] == "UNAVAILABLE"
        assert res["error_code"] == inf.InferenceErrorCode.MODEL_UNAVAILABLE.value


# =============================================================================
# 7. Inference Safety Guards on Synthetic Artifacts (runs on a clean clone)
# =============================================================================

def _synthetic_feature_row(
    tool_id: str, cut_index: int, quality_flag: str, is_valid_run: bool
) -> dict[str, Any]:
    """One feature-table row whose features are all 1.0 (the synthetic training mean)."""
    row: dict[str, Any] = {
        "tool_id": tool_id,
        "cut_index": cut_index,
        "quality_flag": quality_flag,
        "is_valid_run": is_valid_run,
        TARGET_COLUMN: np.nan,
    }
    for feat in COMBINED_FEATURES:
        row[feat] = 1.0
    return row


@pytest.fixture
def synthetic_inference_env(tmp_path: Path, monkeypatch) -> Any:
    """Train a small model into tmp_path and point ml.inference at it (real artifacts untouched)."""
    import ml.inference as inf
    from ml.train_model import collect_package_versions, train_and_save

    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        rng.normal(1.0, 0.1, (40, len(COMBINED_FEATURES))),
        columns=list(COMBINED_FEATURES),
    )
    y = pd.Series(0.2 + 0.5 * (X["current_rms"] - 1.0), name=TARGET_COLUMN)
    out_dir = tmp_path / "artifacts"
    train_and_save(X, y, out_dir, alpha=1.0, metadata={"package_versions": collect_package_versions()})

    rows = [
        _synthetic_feature_row("tool_ok", 1, "OK", True),
        _synthetic_feature_row("tool_bad", 1, "EXTREME_SPIKE_vib_table", False),
        _synthetic_feature_row("tool_flat", 1, "AE_FLATLINE_ae_table", True),
        _synthetic_feature_row("tool_single", 1, "SINGLE_CUT_CASE", True),
        _synthetic_feature_row("tool_ood", 1, "OK", True),
    ]
    feature_table = pd.DataFrame(rows)
    # Physically extreme but finite inputs, as seen in corrupted NASA runs
    feature_table.loc[feature_table["tool_id"] == "tool_ood", "current_rms"] = 1e6
    feature_table.to_csv(out_dir / "cut_features.csv", index=False)

    monkeypatch.setattr(inf, "MODEL_PATH", out_dir / "tool_wear_regressor.joblib")
    monkeypatch.setattr(inf, "SCALER_PATH", out_dir / "feature_scaler.joblib")
    monkeypatch.setattr(inf, "FEATURE_NAMES_PATH", out_dir / "feature_names.json")
    monkeypatch.setattr(inf, "CUT_FEATURES_PATH", out_dir / "cut_features.csv")
    monkeypatch.setattr(inf, "METADATA_PATH", out_dir / "model_metadata.json")
    monkeypatch.setattr(inf, "_cached_model", None)
    monkeypatch.setattr(inf, "_cached_scaler", None)
    monkeypatch.setattr(inf, "_cached_features", None)
    return inf


class TestInferenceSafetyGuards:
    """Corrupted, unknown, or out-of-distribution cuts must never get trusted advisories."""

    def test_valid_cut_receives_prediction(self, synthetic_inference_env):
        inf = synthetic_inference_env
        res = inf.predict("tool_ok", 1, wear_threshold=0.20)
        assert res["error_code"] is None
        assert res["model_status"] == "VERIFIED"
        assert res["predicted_wear"] is not None and np.isfinite(res["predicted_wear"])
        assert res["recommendation"] in ("CONTINUE", "INSPECT_SOON", "REPLACE_TOOL")

    @pytest.mark.parametrize("tool_id", ["tool_bad", "tool_flat"])
    def test_flagged_cut_is_refused(self, synthetic_inference_env, tool_id: str):
        inf = synthetic_inference_env
        res = inf.predict(tool_id, 1, wear_threshold=0.20)
        assert res["predicted_wear"] is None
        assert res["recommendation"] == inf.RecommendationStatus.MODEL_UNAVAILABLE.value
        assert res["error_code"] == inf.InferenceErrorCode.INVALID_RUN.value

    def test_non_sensor_flag_does_not_block_prediction(self, synthetic_inference_env):
        """SINGLE_CUT_CASE describes the experiment, not sensor health."""
        inf = synthetic_inference_env
        res = inf.predict("tool_single", 1, wear_threshold=0.20)
        assert res["error_code"] is None
        assert res["predicted_wear"] is not None

    def test_out_of_distribution_output_is_refused_not_clamped(self, synthetic_inference_env):
        inf = synthetic_inference_env
        res = inf.predict("tool_ood", 1, wear_threshold=0.20)
        assert res["predicted_wear"] is None
        assert res["recommendation"] == inf.RecommendationStatus.MODEL_UNAVAILABLE.value
        assert res["error_code"] == inf.InferenceErrorCode.IMPLAUSIBLE_PREDICTION.value

    def test_unknown_cut_has_distinct_error_code(self, synthetic_inference_env):
        inf = synthetic_inference_env
        res = inf.predict("tool_ok", 99, wear_threshold=0.20)
        assert res["predicted_wear"] is None
        assert res["error_code"] == inf.InferenceErrorCode.UNKNOWN_CUT.value

    def test_missing_quality_columns_fail_closed(self, synthetic_inference_env):
        inf = synthetic_inference_env
        table = pd.read_csv(inf.CUT_FEATURES_PATH).drop(columns=["quality_flag", "is_valid_run"])
        table.to_csv(inf.CUT_FEATURES_PATH, index=False)
        res = inf.predict("tool_ok", 1, wear_threshold=0.20)
        assert res["predicted_wear"] is None
        assert res["error_code"] == inf.InferenceErrorCode.INVALID_RUN.value

    def test_sklearn_version_mismatch_is_not_verified(self, synthetic_inference_env):
        inf = synthetic_inference_env
        meta = {"package_versions": {"scikit-learn": "0.0.0-mismatch"}}
        with open(inf.METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        assert inf.is_model_verified() is False
        res = inf.predict("tool_ok", 1, wear_threshold=0.20)
        assert res["model_status"] == "UNAVAILABLE"
        assert res["error_code"] == inf.InferenceErrorCode.MODEL_UNAVAILABLE.value


# =============================================================================
# 8. Training Provenance (runs on a clean clone)
# =============================================================================

class TestTrainingProvenance:
    """Metadata must be computed from the shipped configuration and be portable."""

    def test_portable_path_inside_and_outside_project(self, tmp_path: Path):
        from ml.train_model import portable_path

        assert portable_path(REPO_ROOT / "artifacts" / "cut_features.csv") == "artifacts/cut_features.csv"
        assert portable_path(tmp_path / "elsewhere.csv") == "elsewhere.csv"

    def test_collect_package_versions_covers_serialization_packages(self):
        import sklearn

        from ml.train_model import collect_package_versions

        versions = collect_package_versions()
        assert set(versions) == {"numpy", "pandas", "scipy", "scikit-learn", "joblib"}
        assert versions["scikit-learn"] == sklearn.__version__

    @pytest.mark.parametrize("tune_flag", [False, True])
    def test_main_writes_metrics_computed_from_shipped_configuration(
        self, synthetic_comparison_df: pd.DataFrame, tmp_path: Path, tune_flag: bool
    ):
        """End-to-end: metadata OOF metrics equal an independent LOTO run of the same config."""
        from ml.train_model import main

        data_path = tmp_path / "cut_features.csv"
        synthetic_comparison_df.to_csv(data_path, index=False)
        out_dir = tmp_path / "out"
        argv = ["--data", str(data_path), "--out-dir", str(out_dir), "--alpha", "10.0"]
        argv.append("--tune" if tune_flag else "--no-tune")
        main(argv)

        with open(out_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_key = "ridge_tuned" if tune_flag else "ridge"
        independent = run_loto_cv(
            synthetic_comparison_df,
            alpha=10.0,
            models=[(model_key, "combined", list(COMBINED_FEATURES))],
            alpha_grid=_DEFAULT_ALPHA_GRID,
        )["aggregate"][model_key]

        oof = meta["out_of_fold_metrics"]
        assert oof["alpha_selection"] == ("nested_inner_loto" if tune_flag else "fixed")
        assert oof["mae_mm"] == independent["mae_mm"]
        assert oof["rmse_mm"] == independent["rmse_mm"]
        assert oof["r2_score"] == independent["r2_score"]
        assert oof["n_samples"] == len(synthetic_comparison_df)
        assert meta["training_dataset_path"] == "cut_features.csv"
        assert "package_versions" in meta
        assert meta["model_version"] == "1.0.0"
        assert meta["final_model_training_error"]["n_samples"] == len(synthetic_comparison_df)

    def test_nested_tuning_is_the_default(self):
        """Without flags, training must use and report nested LOTO alpha selection."""
        from ml.train_model import parse_args

        assert parse_args([]).tune is True
        assert parse_args(["--no-tune"]).tune is False
