"""Custom dataset import routes (Phase 1). Mounted under /api/datasets by backend/main.py.

POST   /api/datasets?filename=…&name=…        raw CSV bytes as the request body (no multipart)
GET    /api/datasets                          built-in NASA entry + imported datasets
GET    /api/datasets/{id}                     manifest, column profile, mapping, validation summary
GET    /api/datasets/{id}/preview             rows as text (offset/limit)
GET    /api/datasets/{id}/mapping/suggestions name-based suggestions (never units)
PUT    /api/datasets/{id}/mapping             save / revise the mapping
POST   /api/datasets/{id}/validate            validation + model-compatibility report
GET    /api/datasets/{id}/validation          latest report
DELETE /api/datasets/{id}                     remove one imported dataset

Errors use HTTP status codes with ``{"detail": {"error_code", "message", "details"}}``.
No route here loads or calls the production model.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.concurrency import run_in_threadpool

from backend import custom_datasets as cd
from backend import custom_models as cm
from backend import training_readiness as tr
from backend.dataset_schemas import (
    CustomModelListResponse,
    CustomModelMetadata,
    CustomModelPredictionsResponse,
    CustomModelPredictRequest,
    CustomModelPredictResponse,
    DatasetDetail,
    DatasetErrorBody,
    DatasetListResponse,
    MappingRecord,
    MappingRequest,
    MappingSuggestion,
    PreviewResponse,
    TrainingReadinessReport,
    ValidationReport,
)

router = APIRouter(prefix="/api/datasets", tags=["Custom datasets"])
_ERRORS = {code: {"model": DatasetErrorBody} for code in (403, 404, 409, 413, 415, 422)}


def _fail(exc: cd.DatasetError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=exc.to_detail())


async def _call(fn, *args):
    try:
        return await run_in_threadpool(fn, *args)
    except cd.DatasetError as exc:
        raise _fail(exc)


@router.post("", response_model=DatasetDetail, status_code=status.HTTP_201_CREATED, responses=_ERRORS,
             summary="Import a CSV file (request body = raw file bytes)")
async def upload_dataset(
    request: Request,
    filename: str = Query(..., min_length=5, max_length=200, description="Original file name; must end in .csv"),
    name: Optional[str] = Query(None, max_length=200, description="Display name"),
):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > cd.MAX_UPLOAD_BYTES:
        raise _fail(cd.DatasetError(413, "FILE_TOO_LARGE", f"The file exceeds the {cd.MAX_UPLOAD_BYTES // (1024 * 1024)} MB import limit.",
                                    {"limit_bytes": cd.MAX_UPLOAD_BYTES}))
    try:
        upload = cd.IncomingUpload(filename, name)
    except cd.DatasetError as exc:
        raise _fail(exc)
    try:
        async for chunk in request.stream():
            upload.write(chunk)
    except cd.DatasetError as exc:
        upload.abort()
        raise _fail(exc)
    except Exception:
        upload.abort()
        raise
    return await _call(upload.finish)


@router.get("", response_model=DatasetListResponse, summary="List the built-in and imported datasets")
async def list_datasets():
    return await _call(cd.list_datasets)


@router.get("/{dataset_id}", response_model=DatasetDetail, responses=_ERRORS, summary="Dataset details and column profile")
async def get_dataset(dataset_id: str):
    return await _call(cd.get_detail, dataset_id)


@router.get("/{dataset_id}/preview", response_model=PreviewResponse, responses=_ERRORS, summary="Rows of the original file, as text")
async def get_preview(dataset_id: str, offset: int = Query(0, ge=0), limit: int = Query(cd.PREVIEW_ROWS, ge=1, le=cd.MAX_PREVIEW_LIMIT)):
    return await _call(cd.get_preview, dataset_id, offset, limit)


@router.get("/{dataset_id}/mapping/suggestions", response_model=MappingSuggestion, responses=_ERRORS,
            summary="Suggested column roles from column names (ambiguous roles left unassigned)")
async def get_mapping_suggestions(dataset_id: str):
    return await _call(cd.suggest_mapping, dataset_id)


@router.put("/{dataset_id}/mapping", response_model=MappingRecord, responses=_ERRORS, summary="Save or revise the column mapping")
async def put_mapping(dataset_id: str, payload: MappingRequest):
    return await _call(cd.save_mapping, dataset_id, payload.model_dump(exclude_none=True))


@router.post("/{dataset_id}/mapping/auto", response_model=MappingRecord, responses=_ERRORS,
             summary="Quick setup: save the suggested mapping for the documented per-cut format, or explain how to fix the CSV (422)")
async def post_auto_mapping(dataset_id: str):
    return await _call(cd.auto_map, dataset_id)


@router.get("/{dataset_id}/progression", responses=_ERRORS,
            summary="Measured wear progression per tool (measured-data analysis; no model is used)")
async def get_progression(dataset_id: str) -> Dict[str, Any]:
    from backend import progression
    return await _call(progression.compute_progression, dataset_id)


@router.post("/{dataset_id}/validate", response_model=ValidationReport, responses=_ERRORS,
             summary="Run validation and the model-compatibility report (never runs the model)")
async def post_validate(dataset_id: str):
    return await _call(cd.validate, dataset_id)


@router.get("/{dataset_id}/validation", response_model=ValidationReport, responses=_ERRORS, summary="Latest validation report")
async def get_validation(dataset_id: str):
    return await _call(cd.get_validation, dataset_id)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT, responses=_ERRORS, summary="Remove one imported dataset")
async def delete_dataset(dataset_id: str):
    await _call(cd.delete_dataset, dataset_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ============================================================================ Phase 2A: Training Readiness, Custom Model Training & Prediction
@router.get("/{dataset_id}/training-readiness", response_model=TrainingReadinessReport, responses=_ERRORS,
            summary="Evaluate whether the dataset is ready for custom model training")
async def get_training_readiness(dataset_id: str):
    return await _call(tr.evaluate_training_readiness, dataset_id)


@router.post("/{dataset_id}/train", response_model=CustomModelMetadata, responses=_ERRORS,
             summary="Train an isolated experimental wear model on this custom dataset")
async def train_model(dataset_id: str, hyperparameters: Optional[Dict[str, Any]] = None):
    return await _call(cm.train_custom_model, dataset_id, hyperparameters)


@router.get("/{dataset_id}/models", response_model=CustomModelListResponse, responses=_ERRORS,
            summary="List all experimental models trained on this dataset")
async def list_models(dataset_id: str):
    return await _call(cm.list_custom_models, dataset_id)


@router.get("/{dataset_id}/models/{model_id}", response_model=CustomModelMetadata, responses=_ERRORS,
            summary="Get details of a specific custom model")
async def get_model(dataset_id: str, model_id: str):
    return await _call(cm.get_custom_model, dataset_id, model_id)


@router.get("/{dataset_id}/models/{model_id}/predictions", response_model=CustomModelPredictionsResponse, responses=_ERRORS,
            summary="Get per-cut predictions for this custom model")
async def get_model_predictions(dataset_id: str, model_id: str):
    return await _call(cm.get_custom_model_predictions, dataset_id, model_id)


@router.post("/{dataset_id}/models/{model_id}/activate", responses=_ERRORS,
             summary="Activate a specific custom model for this dataset")
async def activate_model(dataset_id: str, model_id: str):
    return await _call(cm.activate_custom_model, dataset_id, model_id)


@router.delete("/{dataset_id}/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT, responses=_ERRORS,
               summary="Delete a specific custom model")
async def delete_model(dataset_id: str, model_id: str):
    await _call(cm.delete_custom_model, dataset_id, model_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{dataset_id}/models/{model_id}/predict", response_model=CustomModelPredictResponse, responses=_ERRORS,
             summary="Generate real-time inference using a trained custom model")
async def predict_model(dataset_id: str, model_id: str, body: CustomModelPredictRequest):
    return await _call(cm.predict_custom_model, dataset_id, model_id, body.features)

