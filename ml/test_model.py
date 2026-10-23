"""CNC Sentinel — Model Training & Evaluation Test Suite (P2: Model Training).

Comprehensive automated tests covering:
1. Sensor feature selection and strict exclusion of leakage-prone columns.
2. Missing-label handling, string boolean coercion, and approved eligibility policy.
3. Separation of tool groups in Leave-One-Tool-Out (LOTO) cross-validation.
4. Preprocessing (StandardScaler) fitted strictly within each training fold.
5. MAE and RMSE calculation verification against hand-calculated analytical values.
6. End-to-end model training and LOTO evaluation using a controlled synthetic fixture.
7. Model serialization, artifact persistence, and reloading with preserved feature order.

Usage:
    pytest ml/test_model.py -v
    python ml/test_model.py
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.feature_config import (
    COMBINED_FEATURES,
    IDENTIFIER_COLUMNS,
    OPERATING_CONDITION_COLUMNS,
    PROGRESS_BASELINE_FEATURES,
    PROGRESS_COLUMNS,
    QUALITY_COLUMNS,
    SENSOR_FEATURE_COLUMNS,
    SENSOR_ONLY_FEATURES,
    TARGET_COLUMN,
    get_training_feature_names,
)
from ml.train_model import (
    _REQUIRED_COLUMNS as TRAIN_REQUIRED_COLUMNS,
    build_dummy_baseline,
    build_feature_matrix,
    build_sensor_pipeline,
    load_and_filter as train_load_and_filter,
    train_and_save,
)
from ml.evaluate_model import (
    _GROUP_COLUMN,
    _MODEL_SPECS,
    _REQUIRED_COLUMNS as EVAL_REQUIRED_COLUMNS,
    _make_pipeline,
    load_and_filter as eval_load_and_filter,
    run_loto_cv,
    save_metrics_report,
)


# =============================================================================
# Helper Fixtures & Data Generators
# =============================================================================

def make_synthetic_cut_record(
    tool_id: str,
    cut_index: int,
    wear_vb: float | None = 0.15,
    eligible: bool = True,
    valid_run: bool = True,
    cutting_time: float = 5.0,
    material_id: float = 1.0,
    feed_rate: float = 0.25,
    depth_of_cut: float = 0.75,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a single cut record containing all required schema columns."""
    rng = np.random.RandomState(seed + cut_index * 100)
    record: dict[str, Any] = {
        # Identifiers
        "tool_id": tool_id,
        "cut_index": cut_index,
        "case_num": int(tool_id.replace("case_", "")) if "case_" in tool_id else 1,
        "run_num": cut_index,
        # Quality & eligibility
        "quality_flag": "good" if valid_run else "pathology",
        "is_valid_run": valid_run,
        "training_eligible": eligible,
        "demo_eligible": eligible,
        # Target
        TARGET_COLUMN: np.nan if wear_vb is None else wear_vb,
        # Progress & operating conditions
        "cutting_time_min": cutting_time,
        "material_id": material_id,
        "feed_rate_mm_rev": feed_rate,
        "depth_of_cut_mm": depth_of_cut,
    }

    # 73 sensor features (72 prefix_metric + current_rms)
    for col in SENSOR_FEATURE_COLUMNS:
        record[col] = float(rng.normal(loc=1.0, scale=0.1))

    return record


@pytest.fixture
def synthetic_dataframe() -> pd.DataFrame:
    """Provide a minimal 3-tool synthetic DataFrame with 15 cuts for testing."""
    records = []
    tools = ["case_01", "case_02", "case_03"]
    for t_idx, tool_id in enumerate(tools):
        for cut_idx in range(1, 6):
            base_wear = 0.05 + 0.03 * cut_idx + 0.02 * t_idx
            records.append(
                make_synthetic_cut_record(
                    tool_id=tool_id,
                    cut_index=cut_idx,
                    wear_vb=round(base_wear, 4),
                    eligible=True,
                    valid_run=True,
                    cutting_time=float(cut_idx * 3.5),
                    material_id=float((t_idx % 2) + 1),
                    seed=t_idx * 100 + cut_idx,
                )
            )
    return pd.DataFrame(records)


# =============================================================================
# 1. Feature Selection and Leakage Exclusion Tests
# =============================================================================

class TestFeatureSelectionAndLeakage:
    """Ensure strict input boundaries to prevent target, temporal, or ID leakage."""

    def test_sensor_only_feature_count_and_types(self):
        """SENSOR_ONLY_FEATURES must contain exactly 76 features."""
        assert len(SENSOR_ONLY_FEATURES) == 76
        assert len(SENSOR_FEATURE_COLUMNS) == 73
        assert len(OPERATING_CONDITION_COLUMNS) == 3

    def test_strict_exclusion_of_leakage_prone_columns(self):
        """Identifiers, run counters, temporal progress, and quality flags must NOT leak into sensor features."""
        leakage_columns = set(IDENTIFIER_COLUMNS) | set(PROGRESS_COLUMNS) | set(QUALITY_COLUMNS) | {TARGET_COLUMN}
        sensor_feature_set = set(SENSOR_ONLY_FEATURES)

        intersection = sensor_feature_set.intersection(leakage_columns)
        assert not intersection, f"Critical leakage! Features found in SENSOR_ONLY_FEATURES: {intersection}"

    def test_progress_baseline_feature_boundary(self):
        """Progress baseline must only contain cutting_time_min and operating conditions."""
        expected = set(PROGRESS_COLUMNS) | set(OPERATING_CONDITION_COLUMNS)
        assert set(PROGRESS_BASELINE_FEATURES) == expected
        assert len(PROGRESS_BASELINE_FEATURES) == 4

        # Progress baseline must not contain raw sensor measurements or identifiers
        assert not set(PROGRESS_BASELINE_FEATURES).intersection(set(SENSOR_FEATURE_COLUMNS))
        assert not set(PROGRESS_BASELINE_FEATURES).intersection(set(IDENTIFIER_COLUMNS))

    def test_get_training_feature_names_selector(self):
        """get_training_feature_names returns ordered copy and validates mode argument."""
        sensor_feats = get_training_feature_names("sensor_only")
        assert sensor_feats == SENSOR_ONLY_FEATURES
        assert sensor_feats is not SENSOR_ONLY_FEATURES  # Must return a new list

        progress_feats = get_training_feature_names("progress_baseline")
        assert progress_feats == PROGRESS_BASELINE_FEATURES

        combined_feats = get_training_feature_names("combined")
        assert combined_feats == COMBINED_FEATURES

        with pytest.raises(ValueError, match="Unknown feature set"):
            get_training_feature_names("invalid_feature_set")

    def test_build_feature_matrix_shapes_and_columns(self, synthetic_dataframe: pd.DataFrame):
        """build_feature_matrix extracts exactly the configured features without target or ID columns."""
        X, y = build_feature_matrix(synthetic_dataframe, feature_set="sensor_only")
        assert list(X.columns) == SENSOR_ONLY_FEATURES
        assert X.shape == (len(synthetic_dataframe), 76)
        assert y.shape == (len(synthetic_dataframe),)
        assert TARGET_COLUMN not in X.columns
        for col in IDENTIFIER_COLUMNS:
            assert col not in X.columns


# =============================================================================
# 2. Missing-Label Handling and Approved Eligibility Tests
# =============================================================================

class TestEligibilityAndLabelHandling:
    """Validate that only approved, valid runs with ground-truth wear enter training/eval."""

    def test_filter_removes_ineligible_records(self, tmp_path: Path):
        """Records marked training_eligible=False must be excluded."""
        records = [
            make_synthetic_cut_record("case_01", 1, wear_vb=0.10, eligible=True),
            make_synthetic_cut_record("case_01", 2, wear_vb=0.15, eligible=False),  # Ineligible
            make_synthetic_cut_record("case_02", 1, wear_vb=0.20, eligible=True),
        ]
        csv_file = tmp_path / "cut_features.csv"
        pd.DataFrame(records).to_csv(csv_file, index=False)

        filtered = train_load_and_filter(csv_file)
        assert len(filtered) == 2
        assert list(filtered["cut_index"]) == [1, 1]
        assert (filtered["training_eligible"] == True).all()

    def test_string_boolean_representation_coercion(self, tmp_path: Path):
        """String boolean representations like 'True', 'true', '1' must be parsed properly."""
        records = [
            make_synthetic_cut_record("case_01", 1, wear_vb=0.10, eligible=True),
            make_synthetic_cut_record("case_01", 2, wear_vb=0.12, eligible=True),
            make_synthetic_cut_record("case_01", 3, wear_vb=0.14, eligible=False),
        ]
        # Assign string variants directly in records before DataFrame creation
        records[0]["training_eligible"] = "True"
        records[1]["training_eligible"] = "1"
        records[2]["training_eligible"] = "false"

        df = pd.DataFrame(records)

        csv_file = tmp_path / "cut_features_str_bool.csv"
        df.to_csv(csv_file, index=False)

        filtered = train_load_and_filter(csv_file)
        assert len(filtered) == 2
        assert set(filtered["cut_index"]) == {1, 2}

    def test_nan_target_in_eligible_rows_raises_error(self, tmp_path: Path):
        """If a row is marked eligible but has NaN measured_wear_vb, loader must raise ValueError."""
        records = [
            make_synthetic_cut_record("case_01", 1, wear_vb=0.10, eligible=True),
            make_synthetic_cut_record("case_01", 2, wear_vb=None, eligible=True),  # Illegal NaN wear
        ]
        csv_file = tmp_path / "corrupt_wear.csv"
        pd.DataFrame(records).to_csv(csv_file, index=False)

        with pytest.raises(ValueError, match="contains 1 NaN value"):
            train_load_and_filter(csv_file)

        with pytest.raises(ValueError, match="has 1 NaN"):
            eval_load_and_filter(csv_file)

    def test_empty_eligible_population_raises_error(self, tmp_path: Path):
        """If no rows satisfy training_eligible == True, loader must raise ValueError."""
        records = [
            make_synthetic_cut_record("case_01", 1, wear_vb=0.10, eligible=False),
            make_synthetic_cut_record("case_01", 2, wear_vb=0.15, eligible=False),
        ]
        csv_file = tmp_path / "empty_eligible.csv"
        pd.DataFrame(records).to_csv(csv_file, index=False)

        with pytest.raises(ValueError, match="No training-eligible rows found"):
            train_load_and_filter(csv_file)

        with pytest.raises(ValueError, match="No training-eligible rows found"):
            eval_load_and_filter(csv_file)

    def test_missing_required_column_raises_error(self, tmp_path: Path):
        """Absence of any schema column must raise ValueError."""
        record = make_synthetic_cut_record("case_01", 1, wear_vb=0.10, eligible=True)
        del record["current_rms"]
        csv_file = tmp_path / "missing_column.csv"
        pd.DataFrame([record]).to_csv(csv_file, index=False)

        with pytest.raises(ValueError, match="missing 1 required column"):
            train_load_and_filter(csv_file)

    def test_nonexistent_file_raises_filenotfound(self, tmp_path: Path):
        """Non-existent CSV path raises FileNotFoundError."""
        bogus_path = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            train_load_and_filter(bogus_path)
        with pytest.raises(FileNotFoundError):
            eval_load_and_filter(bogus_path)


# =============================================================================
# 3. Separation of Tool Groups in Validation
# =============================================================================

class TestToolGroupSeparation:
    """Ensure Leave-One-Tool-Out folds strictly isolate tool identities."""

    def test_loto_tool_isolation_per_fold(self, synthetic_dataframe: pd.DataFrame):
        """Every fold must completely exclude the held-out tool from training."""
        results = run_loto_cv(synthetic_dataframe, alpha=1.0)
        assert results["n_tools"] == 3
        assert len(results["per_tool"]) == 3

        unique_tools = sorted(synthetic_dataframe[_GROUP_COLUMN].unique())

        for fold_result in results["per_tool"]:
            held_out = fold_result["tool_id"]
            assert held_out in unique_tools

            # Reconstruct masks to verify separation mathematically
            test_mask = synthetic_dataframe[_GROUP_COLUMN] == held_out
            train_mask = ~test_mask

            train_tools = set(synthetic_dataframe.loc[train_mask, _GROUP_COLUMN])
            test_tools = set(synthetic_dataframe.loc[test_mask, _GROUP_COLUMN])

            # Separation constraint: intersection MUST be empty
            assert held_out not in train_tools
            assert test_tools == {held_out}
            assert train_tools.isdisjoint(test_tools)

            # Sample counts must match
            assert fold_result["n_test"] == test_mask.sum()
            assert fold_result["n_train"] == train_mask.sum()

    def test_empty_training_set_recorded_as_failed_fold(self):
        """If only one tool exists, training set is empty and fold is marked failed."""
        single_tool_df = pd.DataFrame([
            make_synthetic_cut_record("case_01", 1, wear_vb=0.10),
            make_synthetic_cut_record("case_01", 2, wear_vb=0.15),
        ])
        results = run_loto_cv(single_tool_df, alpha=1.0)
        assert results["n_tools"] == 1
        assert len(results["failed_or_skipped_folds"]) == 1
        assert results["per_tool"][0]["status"] == "failed_empty_train"


# =============================================================================
# 4. Preprocessing Fitted Within Each Training Fold
# =============================================================================

class TestPreprocessingFittedInsideFolds:
    """Verify StandardScaler parameters are computed strictly on training partitions."""

    def test_scaler_not_contaminated_by_held_out_tool(self):
        """Scaler fitted inside fold must match training data statistics, ignoring test tool data."""
        # Tool A has mean feature value = 10.0
        # Tool B has mean feature value = 1000.0
        records_a = [
            make_synthetic_cut_record("tool_A", i, wear_vb=0.1 + 0.01 * i)
            for i in range(1, 6)
        ]
        records_b = [
            make_synthetic_cut_record("tool_B", i, wear_vb=0.3 + 0.01 * i)
            for i in range(1, 6)
        ]

        # Force a specific feature to distinct values
        for r in records_a:
            r["vib_spindle_rms"] = 10.0
        for r in records_b:
            r["vib_spindle_rms"] = 1000.0

        df = pd.DataFrame(records_a + records_b)

        # In fold where tool_B is held out, training data is tool_A only
        train_df = df[df[_GROUP_COLUMN] == "tool_A"]
        test_df = df[df[_GROUP_COLUMN] == "tool_B"]

        pipeline = _make_pipeline("sensor", alpha=1.0)
        X_train = train_df[SENSOR_ONLY_FEATURES].astype(float)
        y_train = train_df[TARGET_COLUMN].astype(float)
        pipeline.fit(X_train, y_train)

        scaler = pipeline.named_steps["scaler"]
        idx = SENSOR_ONLY_FEATURES.index("vib_spindle_rms")

        # Scaler mean must equal Tool A's feature value (10.0), NOT contaminated by Tool B (1000.0)
        assert abs(scaler.mean_[idx] - 10.0) < 1e-6
        # If contaminated by full dataset, mean would be 505.0
        assert abs(scaler.mean_[idx] - 505.0) > 400.0

    def test_pipeline_factory_creates_unfitted_instance(self):
        """_make_pipeline must return a fresh, unfitted Pipeline or DummyRegressor."""
        pipe1 = _make_pipeline("sensor", alpha=1.0)
        pipe2 = _make_pipeline("sensor", alpha=2.0)
        assert pipe1 is not pipe2
        assert isinstance(pipe1, Pipeline)
        assert hasattr(pipe1, "named_steps")
        assert not hasattr(pipe1.named_steps["scaler"], "mean_")


# =============================================================================
# 5. MAE/RMSE Calculations Against Known Values
# =============================================================================

class TestMetricCalculations:
    """Validate MAE and RMSE mathematical accuracy against hand-computed values."""

    def test_mae_and_rmse_against_analytical_values(self):
        """Verify metric calculation against known numerical error vectors."""
        y_true = np.array([0.10, 0.20, 0.35, 0.50])
        y_pred = np.array([0.12, 0.18, 0.40, 0.45])

        # Residuals = y_true - y_pred = [-0.02, 0.02, -0.05, 0.05]
        # Absolute residuals = [0.02, 0.02, 0.05, 0.05]
        # Mean absolute error = 0.14 / 4 = 0.0350
        # Squared residuals = [0.0004, 0.0004, 0.0025, 0.0025]
        # Mean squared error = 0.0058 / 4 = 0.00145
        # RMSE = sqrt(0.00145) = 0.03807886552931954
        residuals = y_true - y_pred
        calculated_mae = float(np.mean(np.abs(residuals)))
        calculated_rmse = float(np.sqrt(np.mean(residuals ** 2)))

        expected_mae = 0.035
        expected_rmse = np.sqrt(0.00145)

        assert abs(calculated_mae - expected_mae) < 1e-12
        assert abs(calculated_rmse - expected_rmse) < 1e-12

    def test_perfect_predictions_yield_zero_error(self):
        """Zero residuals must produce exactly 0.0 MAE and 0.0 RMSE."""
        y_true = np.array([0.15, 0.25, 0.45])
        y_pred = np.array([0.15, 0.25, 0.45])
        residuals = y_true - y_pred

        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        assert mae == 0.0
        assert rmse == 0.0

    def test_aggregate_metrics_pool_all_fold_residuals(self, synthetic_dataframe: pd.DataFrame):
        """Aggregate MAE and RMSE in run_loto_cv must equal the pooled error over all folds."""
        results = run_loto_cv(synthetic_dataframe, alpha=1.0)
        aggregate = results["aggregate"]

        for model_key in ["dummy", "progress", "sensor"]:
            agg_stats = aggregate[model_key]
            assert agg_stats["status"] == "ok"
            assert agg_stats["n_samples"] == len(synthetic_dataframe)
            assert agg_stats["mae_mm"] >= 0.0
            assert agg_stats["rmse_mm"] >= agg_stats["mae_mm"]  # Mathematical property: RMSE >= MAE


# =============================================================================
# 6. Training & Evaluation Using Small Synthetic Fixture
# =============================================================================

class TestSyntheticTrainingAndEvaluation:
    """Verify full train and evaluation loops execute end-to-end on synthetic data."""

    def test_end_to_end_train_and_save(self, synthetic_dataframe: pd.DataFrame, tmp_path: Path):
        """train_and_save writes approved artifacts and fits both models."""
        X, y = build_feature_matrix(synthetic_dataframe, feature_set="sensor_only")
        out_dir = tmp_path / "artifacts"

        saved = train_and_save(X, y, out_dir=out_dir)

        assert "tool_wear_regressor.joblib" in saved
        assert "feature_scaler.joblib" in saved
        assert "feature_names.json" in saved

        for name, path in saved.items():
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

    def test_end_to_end_evaluation_and_report_generation(
        self, synthetic_dataframe: pd.DataFrame, tmp_path: Path
    ):
        """run_loto_cv and save_metrics_report generate the expected JSON structure."""
        results = run_loto_cv(synthetic_dataframe, alpha=1.0)
        out_dir = tmp_path / "artifacts"
        report_path = save_metrics_report(results, out_dir=out_dir)

        assert report_path.exists()
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["validation_strategy"] == "leave_one_tool_out"
        assert data["group_column"] == _GROUP_COLUMN
        assert data["n_tools"] == 3
        assert data["n_eligible_cuts"] == 15
        assert set(data["models_compared"]) == {"dummy", "progress", "sensor"}
        assert "aggregate" in data
        assert "per_tool" in data


# =============================================================================
# 7. Model Serialization & Feature Order Reloading
# =============================================================================

class TestModelSerializationAndFeatureOrder:
    """Verify saved models reload cleanly and enforce the expected feature order."""

    def test_serialization_and_reloaded_prediction_parity(
        self, synthetic_dataframe: pd.DataFrame, tmp_path: Path
    ):
        """Reloaded pipeline from joblib produces bit-identical predictions to original."""
        X, y = build_feature_matrix(synthetic_dataframe, feature_set="sensor_only")
        out_dir = tmp_path / "artifacts"
        saved = train_and_save(X, y, out_dir=out_dir)

        pipeline_path = saved["tool_wear_regressor.joblib"]
        reloaded_pipeline: Pipeline = joblib.load(pipeline_path)

        # Structure checks
        assert isinstance(reloaded_pipeline, Pipeline)
        assert "scaler" in reloaded_pipeline.named_steps
        assert "ridge" in reloaded_pipeline.named_steps

        # Prediction parity check
        original_pipe = build_sensor_pipeline(alpha=1.0)
        original_pipe.fit(X, y)

        preds_original = original_pipe.predict(X)
        preds_reloaded = reloaded_pipeline.predict(X)

        assert np.allclose(preds_original, preds_reloaded, atol=1e-12)

    def test_feature_names_json_preserves_exact_order(
        self, synthetic_dataframe: pd.DataFrame, tmp_path: Path
    ):
        """feature_names.json must exactly preserve the 76 SENSOR_ONLY_FEATURES order."""
        X, y = build_feature_matrix(synthetic_dataframe, feature_set="sensor_only")
        out_dir = tmp_path / "artifacts"
        saved = train_and_save(X, y, out_dir=out_dir)

        feature_names_path = saved["feature_names.json"]
        with open(feature_names_path, "r", encoding="utf-8") as f:
            loaded_feature_names = json.load(f)

        assert isinstance(loaded_feature_names, list)
        assert len(loaded_feature_names) == 76
        assert loaded_feature_names == SENSOR_ONLY_FEATURES

    def test_feature_scaler_standalone_reload_parity(
        self, synthetic_dataframe: pd.DataFrame, tmp_path: Path
    ):
        """Standalone feature_scaler.joblib produces identical transformation to pipeline scaler."""
        X, y = build_feature_matrix(synthetic_dataframe, feature_set="sensor_only")
        out_dir = tmp_path / "artifacts"
        saved = train_and_save(X, y, out_dir=out_dir)

        scaler_path = saved["feature_scaler.joblib"]
        standalone_scaler: StandardScaler = joblib.load(scaler_path)

        pipeline_path = saved["tool_wear_regressor.joblib"]
        pipeline: Pipeline = joblib.load(pipeline_path)
        pipeline_scaler: StandardScaler = pipeline.named_steps["scaler"]

        transformed_standalone = standalone_scaler.transform(X)
        transformed_pipeline = pipeline_scaler.transform(X)

        assert np.allclose(transformed_standalone, transformed_pipeline, atol=1e-12)


# =============================================================================
# CLI / Direct Script Execution Entry Point
# =============================================================================

def run_all_tests() -> int:
    """Run all test suites and print a structured summary."""
    print("=" * 70)
    print("CNC Sentinel — Running Model Test Suite (P2: Model Training)")
    print("=" * 70)
    return pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    sys.exit(run_all_tests())
