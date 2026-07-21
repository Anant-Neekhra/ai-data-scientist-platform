"""Dataset upload and validation endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException

from ml_pipeline.data.loader import DatasetLoader, DatasetLoadError
from ml_pipeline.data.validator import DataValidator
from backend.schemas import UploadResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV, validate it, and persist it to the datasets folder."""
    raw_bytes = await file.read()
    try:
        saved_path = DatasetLoader.save_uploaded_file(raw_bytes, file.filename)
        df = DatasetLoader.load_csv(saved_path)
    except DatasetLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))

    validation = DataValidator().validate(df)
    if not validation.is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset failed validation: {validation.errors}",
        )

    return UploadResponse(
        dataset_name=file.filename,
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        validation_errors=validation.errors,
        validation_warnings=validation.warnings,
    )