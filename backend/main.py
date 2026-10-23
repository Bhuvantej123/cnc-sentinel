"""CNC Sentinel - FastAPI Backend Entrypoint.

Minimal skeleton providing health verification, dynamic tool discovery contract,
and model status according to shared/api_contract.md.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure repo root is accessible for shared modules (e.g., ml.inference)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app = FastAPI(
    title="CNC Sentinel API",
    version="0.1.0",
    description="AI-powered CNC cutting-tool wear prediction and RUL estimation API",
)

# Enable CORS for local development
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


@app.get("/api/health", tags=["System"])
async def get_health():
    """Health check endpoint indicating service status and ML artifact readiness.

    Per shared/api_contract.md, model_loaded is true only when the artifact
    is verified to unpickle and satisfy the estimator interface.
    """
    model_loaded = check_model_readiness()

    return {
        "status": "ok",
        "service": "cnc-sentinel-backend",
        "version": "0.1.0",
        "model_loaded": model_loaded,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
