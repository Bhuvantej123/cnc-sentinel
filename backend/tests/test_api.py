"""Integration tests for all CNC Sentinel backend API endpoints.

Tests:
1. GET  /api/health (service status and model verification readiness)
2. GET  /api/meta (system metadata, threshold limits, sensor channels)
3. GET  /api/tools (dynamic discovery of the 16 NASA Milling cases)
4. GET  /api/tools/{tool_id}/history (preservation of physical runs and nulls, 404 on missing)
5. POST /api/predict (validation, safe fallback to MODEL_UNAVAILABLE, 404, and 422 validations)
6. GET  /api/generalization (safe baseline PENDING_VERIFICATION)
"""

from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main import app

client = TestClient(app)


# =========================================================================
# 1. Health Endpoint Tests
# =========================================================================

def test_get_health():
    """Verify that GET /api/health returns 200, valid service name, and model_loaded=False."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "cnc-sentinel-backend"
    assert data["version"] == "0.1.0"
    assert data["model_loaded"] is False
    assert "timestamp" in data


# =========================================================================
# 2. Metadata Endpoint Tests
# =========================================================================

def test_get_meta():
    """Verify system metadata and configuration parameters."""
    response = client.get("/api/meta")
    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "CNC Sentinel"
    assert data["primary_dataset"] == "NASA Milling Dataset"
    assert data["wear_metric"] == "flank_wear_vb"
    assert data["wear_unit"] == "mm"
    assert data["default_wear_threshold"] == 0.20
    assert data["max_allowable_wear"] == 0.30
    assert set(data["sensors_supported"]) == {
        "smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"
    }
    assert set(data["risk_tolerance_levels"]) == {"conservative", "balanced", "aggressive"}
    assert data["model_info"]["status"] == "PENDING_VERIFICATION"
    assert data["model_info"]["feature_count"] == 0


# =========================================================================
# 3. Tools Discovery Endpoint Tests
# =========================================================================

def test_get_tools():
    """Verify dynamic discovery of all 16 tool cases from cut_features.csv."""
    response = client.get("/api/tools")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    tools = data["tools"]
    assert len(tools) == 16

    # Verify tool identifiers are formatted as case_01 .. case_16
    tool_ids = [t["tool_id"] for t in tools]
    assert tool_ids == [f"case_{i:02d}" for i in range(1, 17)]

    # Check Case 1 specs
    case_1 = next(t for t in tools if t["tool_id"] == "case_01")
    assert case_1["material_id"] == 1
    assert case_1["feed_rate_mm_rev"] == 0.5
    assert case_1["depth_of_cut_mm"] == 1.5
    assert case_1["total_recorded_cuts"] == 17
    assert case_1["last_cut_index"] == 17
    assert case_1["status"] == "ACTIVE"

    # Check Case 3 specs (preserving non-consecutive physical runs)
    case_3 = next(t for t in tools if t["tool_id"] == "case_03")
    assert case_3["total_recorded_cuts"] == 14
    assert case_3["last_cut_index"] == 16


# =========================================================================
# 4. Tool History Endpoint Tests
# =========================================================================

def test_get_tool_history_success():
    """Verify tool history accurately reports physical cut indices, measured values, and nulls."""
    response = client.get("/api/tools/case_01/history")
    assert response.status_code == 200
    data = response.json()
    assert data["tool_id"] == "case_01"
    assert data["model_status"] == "UNAVAILABLE"
    history = data["history"]
    assert len(history) == 17

    # Cut 1 had wear measured (0.00 mm)
    cut_1 = history[0]
    assert cut_1["cut_index"] == 1
    assert cut_1["measured_wear"] == 0.0
    assert cut_1["predicted_wear"] is None
    assert cut_1["uncertainty_interval"] is None

    # Cut 2 had missing wear (should be null, not 0.0 or interpolated)
    cut_2 = history[1]
    assert cut_2["cut_index"] == 2
    assert cut_2["measured_wear"] is None
    assert cut_2["predicted_wear"] is None


def test_get_tool_history_non_consecutive():
    """Verify Case 3 history preserves non-consecutive run indices [1,2,3,5..16]."""
    response = client.get("/api/tools/case_03/history")
    assert response.status_code == 200
    data = response.json()
    history = data["history"]
    indices = [h["cut_index"] for h in history]
    assert indices == [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]


def test_get_tool_history_not_found():
    """Verify 404 when querying an unregistered tool case."""
    response = client.get("/api/tools/case_99/history")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# =========================================================================
# 5. Prediction Endpoint Tests
# =========================================================================

def test_post_predict_valid_unverified_model():
    """Verify POST /api/predict safely returns MODEL_UNAVAILABLE with nulls while weights are unverified."""
    payload = {
        "tool_id": "case_01",
        "cut_index": 5,
        "wear_threshold": 0.20,
        "risk_tolerance": "balanced",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tool_id"] == "case_01"
    assert data["cut_index"] == 5
    assert data["predicted_wear"] is None
    assert data["wear_unit"] == "mm"
    assert data["uncertainty_interval"] is None
    assert data["estimated_rul_cuts"] is None
    assert data["recommendation"] == "MODEL_UNAVAILABLE"
    assert data["model_status"] == "UNAVAILABLE"
    assert "unavailable" in data["explanation"].lower()


def test_post_predict_tool_not_found():
    """Verify 404 when predicting on an unknown tool ID."""
    payload = {
        "tool_id": "case_unknown",
        "cut_index": 1,
        "wear_threshold": 0.20,
        "risk_tolerance": "balanced",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_post_predict_invalid_wear_threshold():
    """Verify 422 when wear_threshold is <= 0."""
    payload = {
        "tool_id": "case_01",
        "cut_index": 1,
        "wear_threshold": -0.05,
        "risk_tolerance": "balanced",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_post_predict_invalid_cut_index():
    """Verify 422 when cut_index is < 1."""
    payload = {
        "tool_id": "case_01",
        "cut_index": 0,
        "wear_threshold": 0.20,
        "risk_tolerance": "balanced",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_post_predict_invalid_risk_tolerance():
    """Verify 422 when risk_tolerance is not in allowed enum."""
    payload = {
        "tool_id": "case_01",
        "cut_index": 1,
        "wear_threshold": 0.20,
        "risk_tolerance": "extreme_risk",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


# =========================================================================
# 6. Generalization Endpoint Tests
# =========================================================================

def test_get_generalization_pending():
    """Verify GET /api/generalization returns safe baseline PENDING_VERIFICATION response."""
    response = client.get("/api/generalization")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PENDING_VERIFICATION"
    assert data["evaluation_strategy"] == "Leave-One-Tool-Out (LOTO)"
    assert data["metrics"]["r2_score"] is None
    assert data["metrics"]["rmse_mm"] is None
    assert data["metrics"]["mae_mm"] is None
    assert data["metrics"]["rul_mean_error_cuts"] is None
    assert "pending" in data["notes"].lower() or "validates" in data["notes"].lower()
