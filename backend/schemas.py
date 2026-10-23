"""Pydantic schemas matching shared/api_contract.md specification."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = "ok"
    service: str = "cnc-sentinel-backend"
    version: str = "0.1.0"
    model_loaded: bool = False
    timestamp: str


class ModelInfo(BaseModel):
    """Model metadata block inside MetaResponse."""
    model_name: str = "WearRegressionRUL"
    status: Literal["PENDING_VERIFICATION", "VERIFIED", "UNAVAILABLE"] = "PENDING_VERIFICATION"
    trained_at: Optional[str] = None
    feature_count: int = 0


class MetaResponse(BaseModel):
    """System metadata and configuration schema."""
    project_name: str = "CNC Sentinel"
    primary_dataset: str = "NASA Milling Dataset"
    wear_metric: str = "flank_wear_vb"
    wear_unit: str = "mm"
    default_wear_threshold: float = 0.20
    max_allowable_wear: float = 0.30
    sensors_supported: List[str] = [
        "smcAC",
        "smcDC",
        "vib_table",
        "vib_spindle",
        "AE_table",
        "AE_spindle",
    ]
    risk_tolerance_levels: List[str] = [
        "conservative",
        "balanced",
        "aggressive",
    ]
    model_info: ModelInfo


class ToolSummary(BaseModel):
    """Summary of a cutting tool case discovered from the dataset."""
    tool_id: str
    material_id: Optional[int] = None
    feed_rate_mm_rev: Optional[float] = None
    depth_of_cut_mm: Optional[float] = None
    total_recorded_cuts: int
    last_cut_index: int
    status: str = "ACTIVE"


class ToolsResponse(BaseModel):
    """Response containing list of registered/discovered tools."""
    tools: List[ToolSummary]


class UncertaintyInterval(BaseModel):
    """Confidence bound container."""
    lower: float
    upper: float
    confidence_level: float = 0.95


class ToolHistoryItem(BaseModel):
    """Single cut point in tool history."""
    cut_index: int
    measured_wear: Optional[float] = None
    predicted_wear: Optional[float] = None
    uncertainty_interval: Optional[UncertaintyInterval] = None


class ToolHistoryResponse(BaseModel):
    """Cut-by-cut wear progression history for a tool."""
    tool_id: str
    history: List[ToolHistoryItem]
    model_status: Literal["VERIFIED", "UNAVAILABLE", "PENDING_VERIFICATION"] = "UNAVAILABLE"


class PredictRequest(BaseModel):
    """Request payload for wear prediction and RUL estimation."""
    tool_id: str
    cut_index: int = Field(..., ge=1, description="1-indexed current cut index")
    wear_threshold: float = Field(..., gt=0.0, description="Wear limit in mm (must be positive)")
    risk_tolerance: Literal["conservative", "balanced", "aggressive"] = "balanced"

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("tool_id cannot be empty")
        return v_stripped


class PredictResponse(BaseModel):
    """Prediction, uncertainty, RUL, and advisory response."""
    tool_id: str
    cut_index: int
    predicted_wear: Optional[float] = None
    wear_unit: str = "mm"
    uncertainty_interval: Optional[UncertaintyInterval] = None
    estimated_rul_cuts: Optional[int] = None
    recommendation: Literal["CONTINUE", "INSPECT_SOON", "REPLACE_TOOL", "MODEL_UNAVAILABLE"]
    explanation: str
    model_status: Literal["VERIFIED", "UNAVAILABLE", "PENDING_VERIFICATION"]


class GeneralizationMetrics(BaseModel):
    """Cross-validation performance metric scores."""
    r2_score: Optional[float] = None
    rmse_mm: Optional[float] = None
    mae_mm: Optional[float] = None
    rul_mean_error_cuts: Optional[float] = None


class GeneralizationResponse(BaseModel):
    """Response for Leave-One-Tool-Out cross validation."""
    status: Literal["PENDING_VERIFICATION", "VERIFIED"]
    evaluation_strategy: str = "Leave-One-Tool-Out (LOTO)"
    metrics: GeneralizationMetrics
    notes: str
