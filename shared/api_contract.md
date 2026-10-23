# CNC Sentinel — API Contract Specification

This document defines the interface contract between the **FastAPI Backend** and the **React + Vite Frontend**. Both teams must strictly adhere to these schemas to enable parallel development.

---

## 1. Global Conventions

- **Base URL**: `http://localhost:8000/api` (or `/api` via Vite dev proxy)
- **Content-Type**: `application/json`
- **Verification Rule**: All model outputs and accuracy metrics are flagged as `UNAVAILABLE` or `null` until a genuine, validated model artifact is loaded and verified. No fabricated predictions or mock performance values are permitted from the backend.
- **Dynamic Discovery**: Tool entities and available cuts are discovered dynamically from verified artifacts (`artifacts/cut_features.csv`). The backend does not serve hard-coded invented tools.

---

## 2. Endpoints

### 2.1 GET `/api/health`
Checks backend service availability and whether a verified ML model artifact has passed runtime validation.

> [!NOTE]
> A file merely existing on disk does **not** set `model_loaded: true`. The backend verifies the estimator interface and feature schema at startup.

#### Response: `200 OK`
```json
{
  "status": "ok",
  "service": "cnc-sentinel-backend",
  "version": "0.1.0",
  "model_loaded": false,
  "timestamp": "2026-10-23T10:45:00Z"
}
```

---

### 2.2 GET `/api/meta`
Returns system metadata, operational wear thresholds, unit standards, and dataset configuration parameters.

#### Response: `200 OK`
```json
{
  "project_name": "CNC Sentinel",
  "primary_dataset": "NASA Milling Dataset",
  "wear_metric": "flank_wear_vb",
  "wear_unit": "mm",
  "default_wear_threshold": 0.20,
  "max_allowable_wear": 0.30,
  "sensors_supported": [
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle"
  ],
  "risk_tolerance_levels": [
    "conservative",
    "balanced",
    "aggressive"
  ],
  "model_info": {
    "model_name": "WearRegressionRUL",
    "status": "PENDING_VERIFICATION",
    "trained_at": null,
    "feature_count": 0
  }
}
```

---

### 2.3 GET `/api/tools`
Lists tracked cutting tools discovered dynamically from `artifacts/cut_features.csv`. If no feature table has been generated, returns an empty array.

#### Response: `200 OK` (When `artifacts/cut_features.csv` is populated)
```json
{
  "tools": [
    {
      "tool_id": "case_01",
      "material_id": 1,
      "feed_rate_mm_rev": 0.5,
      "depth_of_cut_mm": 1.5,
      "total_recorded_cuts": 17,
      "last_cut_index": 17,
      "status": "ACTIVE"
    },
    {
      "tool_id": "case_02",
      "material_id": 2,
      "feed_rate_mm_rev": 0.25,
      "depth_of_cut_mm": 0.75,
      "total_recorded_cuts": 11,
      "last_cut_index": 11,
      "status": "ACTIVE"
    }
  ]
}
```

#### Response: `200 OK` (Before artifacts generation)
```json
{
  "tools": []
}
```

---

### 2.4 GET `/api/tools/{tool_id}/history`
Retrieves cut-by-cut wear history for a given tool case.

> [!IMPORTANT]
> **Distinguishing Measured Wear from Predictions**:
> In datasets like NASA Milling, flank wear was measured intermittently via optical microscope.
> - Cuts where wear was physically measured contain a float in `measured_wear`.
> - Cuts where no physical measurement was taken contain `measured_wear: null`.
> - Unpredicted cuts or runs where the model is unverified contain `predicted_wear: null` and `uncertainty_interval: null`.
> - Missing values must remain `null`—no interpolation or fake values.

#### Response: `200 OK`
```json
{
  "tool_id": "case_01",
  "history": [
    {
      "cut_index": 1,
      "measured_wear": 0.051,
      "predicted_wear": null,
      "uncertainty_interval": null
    },
    {
      "cut_index": 2,
      "measured_wear": null,
      "predicted_wear": null,
      "uncertainty_interval": null
    },
    {
      "cut_index": 3,
      "measured_wear": 0.082,
      "predicted_wear": null,
      "uncertainty_interval": null
    }
  ],
  "model_status": "UNAVAILABLE"
}
```

---

### 2.5 POST `/api/predict`
Calculates cutting-tool flank wear prediction, uncertainty interval, and remaining useful life (RUL) in cuts based on the current cut state and operator risk tolerance.

All prediction and RUL logic is owned by `ml/inference.py`. The backend validates the HTTP request and delegates to `ml.inference.predict()`.

#### Cut Index Semantics:
`cut_index` represents the **current recorded cut index** of the tool run. The model predicts the flank wear for this current pass and projects Remaining Useful Life (RUL)—the number of additional safe cutting passes before flank wear reaches `wear_threshold`.

#### Risk Tolerance Definitions:
- `"conservative"`: Evaluates RUL against the upper uncertainty bound (e.g. 95% upper limit), flagging earlier tool retirement.
- `"balanced"`: Evaluates RUL against expected mean wear point prediction.
- `"aggressive"`: Evaluates RUL against the lower uncertainty bound, tolerating more cutting passes.
- *Notice*: Confidence interval calibration must be empirically tested on holdout cuts before calibration is claimed.

#### Request Body
```json
{
  "tool_id": "case_01",
  "cut_index": 12,
  "wear_threshold": 0.20,
  "risk_tolerance": "balanced"
}
```

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `tool_id` | `string` | Yes | Identifier for the tool or cutting case (e.g., `"case_01"`). |
| `cut_index` | `integer` | Yes | The current recorded cut index ($1 \le \text{cut\_index} \le \text{max\_cuts}$). |
| `wear_threshold` | `float` | Yes | Flank wear limit ($V_B$ in mm) defining tool retirement (e.g. `0.20`). Must be verified against dataset. |
| `risk_tolerance` | `string` | Yes | `"conservative"`, `"balanced"`, or `"aggressive"`. |

#### Response: `200 OK` (When ML model artifact is loaded and verified)
```json
{
  "tool_id": "case_01",
  "cut_index": 12,
  "predicted_wear": 0.142,
  "wear_unit": "mm",
  "uncertainty_interval": {
    "lower": 0.128,
    "upper": 0.156,
    "confidence_level": 0.95
  },
  "estimated_rul_cuts": 5,
  "recommendation": "CONTINUE",
  "explanation": "Predicted flank wear is 0.142 mm, below the 0.20 mm limit. At current wear rate, approximately 5 safe cuts remain.",
  "model_status": "VERIFIED"
}
```

#### Response: `200 OK` (Baseline State — Before Verification)
```json
{
  "tool_id": "case_01",
  "cut_index": 12,
  "predicted_wear": null,
  "wear_unit": "mm",
  "uncertainty_interval": null,
  "estimated_rul_cuts": null,
  "recommendation": "MODEL_UNAVAILABLE",
  "explanation": "ML model weights or verified feature table (cut_features.csv) have not yet been generated from the NASA Milling dataset. Predictions and RUL are currently unavailable.",
  "model_status": "UNAVAILABLE"
}
```

#### Recommendation Status Enum:
- `"CONTINUE"`: Wear is safely beneath the warning threshold.
- `"INSPECT_SOON"`: Wear has entered the precautionary zone ($\ge 80\%$ of threshold); inspection advised.
- `"REPLACE_TOOL"`: Wear has reached or is projected to exceed the threshold in the next pass.
- `"MODEL_UNAVAILABLE"`: Model or required evidence is missing/unverified.

---

### 2.6 GET `/api/generalization`
Returns Leave-One-Tool-Out (LOTO) cross-validation metrics across unseen tools from the NASA Milling runs.

#### Response: `200 OK` (Baseline / Pending State)
```json
{
  "status": "PENDING_VERIFICATION",
  "evaluation_strategy": "Leave-One-Tool-Out (LOTO)",
  "metrics": {
    "r2_score": null,
    "rmse_mm": null,
    "mae_mm": null,
    "rul_mean_error_cuts": null
  },
  "notes": "Generalization metrics will be populated once ml/train.py validates against the NASA Milling dataset."
}
```

---

## 3. Error Responses

```json
{
  "detail": "Description of the validation or processing error"
}
```

- `400 Bad Request`: Invalid parameters (e.g., negative cut index, invalid wear threshold).
- `404 Not Found`: Tool ID does not exist in the discovered feature table.
- `422 Unprocessable Entity`: Request body failed Pydantic schema validation.
- `500 Internal Server Error`: Server failure.
