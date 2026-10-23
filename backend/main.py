"""CNC Sentinel - FastAPI Backend Entrypoint.

Implements all endpoints defined in shared/api_contract.md:
- GET  /api/health
- GET  /api/meta
- GET  /api/tools
- GET  /api/tools/{tool_id}/history
- POST /api/predict
- GET  /api/generalization
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure repo root is accessible for shared modules (e.g., ml.inference)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.schemas import (
    HealthResponse,
    MetaResponse,
    ToolsResponse,
    ToolHistoryResponse,
    PredictRequest,
    PredictResponse,
    GeneralizationResponse,
)
from backend import services

app = FastAPI(
    title="CNC Sentinel API",
    version="0.1.0",
    description="AI-powered CNC cutting-tool wear prediction and RUL estimation API",
)

# Enable CORS for local Vite development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_model_readiness() -> bool:
    """Verify model readiness through runtime validation, not merely file presence.

    Delegates to ml.inference.is_model_verified() if available.
    """
    try:
        from ml.inference import is_model_verified
        return is_model_verified()
    except Exception:
        return False


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Health & Model Readiness Check",
)
async def get_health():
    """Health check endpoint indicating service status and ML artifact readiness.

    Per shared/api_contract.md, model_loaded is true only when the artifact
    is verified to unpickle and satisfy the estimator interface.
    """
    model_loaded = check_model_readiness()

    return HealthResponse(
        status="ok",
        service="cnc-sentinel-backend",
        version="0.1.0",
        model_loaded=model_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/api/meta",
    response_model=MetaResponse,
    tags=["System"],
    summary="System Metadata & Operational Thresholds",
)
async def get_meta():
    """Returns system configuration, wear units, operational limits, and supported sensor channels."""
    return services.get_system_meta()


@app.get(
    "/api/tools",
    response_model=ToolsResponse,
    tags=["Tools"],
    summary="List Tracked Cutting Tools",
)
async def get_tools():
    """Lists cutting tool cases discovered dynamically from artifacts/cut_features.csv."""
    tools = services.get_discovered_tools()
    return ToolsResponse(tools=tools)


@app.get(
    "/api/tools/{tool_id}/history",
    response_model=ToolHistoryResponse,
    tags=["Tools"],
    summary="Get Tool Wear Progression History",
)
async def get_tool_history(tool_id: str):
    """Retrieves cut-by-cut wear progression history for a specific tool.

    Missing optical microscope measurements remain null. Returns 404 if tool does not exist.
    """
    history_resp = services.get_tool_history(tool_id)
    if history_resp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_id}' not found in registry.",
        )
    return history_resp


@app.post(
    "/api/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Predict Tool Wear and Remaining Useful Life",
)
async def post_predict(payload: PredictRequest):
    """Calculates cutting-tool flank wear prediction, uncertainty interval, and RUL.

    Delegates calculation directly to ml.inference.predict().
    Returns safe MODEL_UNAVAILABLE state with null predictions if model is pending verification.
    """
    # Verify tool exists in registry
    if not services.tool_exists(payload.tool_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{payload.tool_id}' not found in registry.",
        )

    try:
        from ml.inference import predict
        result = predict(
            tool_id=payload.tool_id,
            cut_index=payload.cut_index,
            wear_threshold=payload.wear_threshold,
            risk_tolerance=payload.risk_tolerance,
        )
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@app.get(
    "/api/generalization",
    response_model=GeneralizationResponse,
    tags=["Inference"],
    summary="Cross-Tool Validation Metrics",
)
async def get_generalization():
    """Returns Leave-One-Tool-Out (LOTO) generalization metrics.

    Returns PENDING_VERIFICATION until real models are evaluated.
    """
    return services.get_generalization_report()
