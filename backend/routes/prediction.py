"""
Prediction endpoints: single prediction (JSON in, JSON out) and
batch prediction (CSV upload in, CSV download out).
"""

import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from ml_pipeline.prediction.predictor import PredictionService, PredictionInputError
from backend.schemas import SinglePredictionRequest, PredictionResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
_service = PredictionService()


@router.post("/predict", response_model=PredictionResponse)
def predict_single(request: SinglePredictionRequest):
    """Predict on a single row of feature values."""
    try:
        result = _service.predict_single(request.dataset_name, request.row, request.version)
    except PredictionInputError as e:
        # Client sent bad/incomplete data -> 400, not 500. Distinguishing
        # "you made a mistake" (400) from "we made a mistake" (500) is
        # standard REST practice and matters for anyone building
        # against this API later.
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return PredictionResponse(
        predictions=result.predictions,
        probabilities=result.probabilities,
        model_name=result.model_name,
        model_version=result.model_version,
    )


@router.post("/predict/batch")
def predict_batch(
    dataset_name: str,
    file: UploadFile = File(...),
    version: int | None = None,
):
    """
    Upload a CSV, get back a CSV with predictions appended, streamed
    directly as a file download.
    """
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse uploaded CSV: {e}")

    try:
        result_df = _service.predict_batch(dataset_name, df, version)
    except PredictionInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Write to an in-memory buffer rather than a temp file on disk --
    # this response never needs to persist, so streaming from memory
    # avoids filesystem cleanup entirely.
    buffer = io.StringIO()
    result_df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )