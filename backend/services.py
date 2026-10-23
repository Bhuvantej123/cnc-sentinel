"""Backend services for feature table loading, tool discovery, and history."""

from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import numpy as np
import pandas as pd

from backend.schemas import (
    ToolSummary,
    ToolsResponse,
    ToolHistoryItem,
    ToolHistoryResponse,
    GeneralizationMetrics,
    GeneralizationResponse,
    ModelInfo,
    MetaResponse,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
CUT_FEATURES_PATH = ARTIFACTS_DIR / "cut_features.csv"
METRICS_REPORT_PATH = ARTIFACTS_DIR / "metrics_report.json"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"


def load_cut_features() -> Optional[pd.DataFrame]:
    """Safely load the verified cut features table."""
    if not CUT_FEATURES_PATH.exists():
        return None
    try:
        df = pd.read_csv(CUT_FEATURES_PATH)
        required_cols = {"tool_id", "cut_index", "material_id", "feed_rate_mm_rev", "depth_of_cut_mm", "measured_wear_vb"}
        if not required_cols.issubset(set(df.columns)):
            return None
        return df
    except Exception:
        return None


def get_discovered_tools() -> List[ToolSummary]:
    """Dynamically discover cutting tools and operating conditions from cut_features.csv."""
    df = load_cut_features()
    if df is None or len(df) == 0:
        return []

    tools: List[ToolSummary] = []
    # Sort tools naturally by case number if available, else by tool_id
    tool_groups = df.groupby("tool_id", sort=False)
    # Sort keys for consistent ordering: case_01, case_02, ...
    sorted_tool_ids = sorted(df["tool_id"].unique())

    for tool_id in sorted_tool_ids:
        group = df[df["tool_id"] == tool_id].sort_values("cut_index")
        first_row = group.iloc[0]

        material_id = int(first_row["material_id"]) if pd.notna(first_row["material_id"]) else None
        feed_rate = float(first_row["feed_rate_mm_rev"]) if pd.notna(first_row["feed_rate_mm_rev"]) else None
        depth_of_cut = float(first_row["depth_of_cut_mm"]) if pd.notna(first_row["depth_of_cut_mm"]) else None

        tools.append(
            ToolSummary(
                tool_id=str(tool_id),
                material_id=material_id,
                feed_rate_mm_rev=feed_rate,
                depth_of_cut_mm=depth_of_cut,
                total_recorded_cuts=len(group),
                last_cut_index=int(group["cut_index"].max()),
                status="ACTIVE",
            )
        )

    return tools


def tool_exists(tool_id: str) -> bool:
    """Check if a tool exists in the discovered dataset."""
    df = load_cut_features()
    if df is None:
        return False
    return tool_id in df["tool_id"].values


def get_tool_history(tool_id: str) -> Optional[ToolHistoryResponse]:
    """Build cut-by-cut wear history for a specific tool.

    Missing wear measurements remain null; physical cut indices are strictly preserved.
    """
    df = load_cut_features()
    if df is None:
        return None

    tool_df = df[df["tool_id"] == tool_id].sort_values("cut_index")
    if len(tool_df) == 0:
        return None

    history_items: List[ToolHistoryItem] = []
    for _, row in tool_df.iterrows():
        raw_wear = row["measured_wear_vb"]
        measured_wear: Optional[float] = None
        if pd.notna(raw_wear) and np.isfinite(raw_wear):
            measured_wear = round(float(raw_wear), 4)

        history_items.append(
            ToolHistoryItem(
                cut_index=int(row["cut_index"]),
                measured_wear=measured_wear,
                predicted_wear=None,
                uncertainty_interval=None,
            )
        )

    # Check model status
    try:
        from ml.inference import is_model_verified
        model_status = "VERIFIED" if is_model_verified() else "UNAVAILABLE"
    except Exception:
        model_status = "UNAVAILABLE"

    return ToolHistoryResponse(
        tool_id=tool_id,
        history=history_items,
        model_status=model_status,
    )


def get_system_meta() -> MetaResponse:
    """Build system metadata and model verification status."""
    feature_count = 0
    trained_at = None
    model_status: str = "PENDING_VERIFICATION"

    try:
        from ml.inference import is_model_verified
        if is_model_verified():
            model_status = "VERIFIED"
            if FEATURE_NAMES_PATH.exists():
                with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
                    features = json.load(f)
                    feature_count = len(features) if isinstance(features, list) else 0
    except Exception:
        model_status = "PENDING_VERIFICATION"

    return MetaResponse(
        project_name="CNC Sentinel",
        primary_dataset="NASA Milling Dataset",
        wear_metric="flank_wear_vb",
        wear_unit="mm",
        default_wear_threshold=0.20,
        max_allowable_wear=0.30,
        sensors_supported=[
            "smcAC",
            "smcDC",
            "vib_table",
            "vib_spindle",
            "AE_table",
            "AE_spindle",
        ],
        risk_tolerance_levels=[
            "conservative",
            "balanced",
            "aggressive",
        ],
        model_info=ModelInfo(
            model_name="WearRegressionRUL",
            status=model_status,
            trained_at=trained_at,
            feature_count=feature_count,
        ),
    )


def get_generalization_report() -> GeneralizationResponse:
    """Retrieve generalization metrics from artifacts/metrics_report.json.

    Returns safe baseline PENDING_VERIFICATION response if artifact is absent.
    """
    if METRICS_REPORT_PATH.exists():
        try:
            with open(METRICS_REPORT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            metrics_data = data.get("metrics", {})
            return GeneralizationResponse(
                status="VERIFIED" if data.get("status") == "VERIFIED" else "PENDING_VERIFICATION",
                evaluation_strategy=data.get("evaluation_strategy", "Leave-One-Tool-Out (LOTO)"),
                metrics=GeneralizationMetrics(
                    r2_score=metrics_data.get("r2_score"),
                    rmse_mm=metrics_data.get("rmse_mm"),
                    mae_mm=metrics_data.get("mae_mm"),
                    rul_mean_error_cuts=metrics_data.get("rul_mean_error_cuts"),
                ),
                notes=data.get("notes", "Computed over unseen test tools."),
            )
        except Exception:
            pass

    return GeneralizationResponse(
        status="PENDING_VERIFICATION",
        evaluation_strategy="Leave-One-Tool-Out (LOTO)",
        metrics=GeneralizationMetrics(
            r2_score=None,
            rmse_mm=None,
            mae_mm=None,
            rul_mean_error_cuts=None,
        ),
        notes="Generalization metrics will be populated once ml/train.py validates against the NASA Milling dataset.",
    )
