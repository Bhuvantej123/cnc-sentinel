# Backend — CNC Sentinel

FastAPI service powering the CNC Sentinel inference and management API.

---

## Assigned Contributor: P3 (Backend Developer)

### Responsibilities:
1. Implement the API endpoints specified in [`../shared/api_contract.md`](../shared/api_contract.md):
   - `GET /api/health`: Health status & verified model check (already implemented in `main.py`).
   - `GET /api/meta`: System metadata and operational limits.
   - `GET /api/tools`: Dynamically discovers tool cases from `artifacts/cut_features.csv` (no hard-coded tools).
   - `GET /api/tools/{tool_id}/history`: Returns cut-by-cut measured wear and prediction slots. Distinguishes measured from predicted; missing values remain `null`.
   - `POST /api/predict`: Validates input and delegates prediction directly to `ml.inference.predict()`.
   - `GET /api/generalization`: Reads evaluation metrics from `artifacts/metrics_report.json`.

---

## Model Loading & Dependency Strategy

- **Single Inference Interface**: The backend does **not** implement custom inference or RUL arithmetic. It imports `ml.inference.predict()` and `ml.inference.validate_model_artifact()`.
- **Runtime Dependencies**: Pinned in `requirements.txt` to match `ml/requirements.txt` (`scikit-learn==1.3.2`, `numpy==1.26.2`, `pandas==2.1.4`, `joblib==1.3.2`). This ensures that unpickling serialized scikit-learn models is safe and free from ABI/version incompatibilities.
- **Verification Rule**: A file existing on disk does not set `model_loaded=true`. The artifact is validated at startup via `ml.inference.validate_model_artifact()`.

---

## Local Development Commands

```bash
# In backend/ directory:
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Interactive API documentation will be available at:
`http://localhost:8000/docs`
