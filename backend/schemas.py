"""
Pydantic request/response models for the API. Unlike the dataclasses
used throughout ml_pipeline/, these VALIDATE incoming data at
construction time -- exactly what's needed at an HTTP boundary,
where input can't be trusted the way an internal function call can.
"""

from pydantic import BaseModel, Field


class SinglePredictionRequest(BaseModel):
    """Request body for POST /predict -- arbitrary feature columns."""

    dataset_name: str = Field(..., description="Which dataset's model to use, e.g. 'titanic.csv'")
    row: dict = Field(..., description="Feature values as {column_name: value}")
    version: int | None = Field(None, description="Specific model version; omit for current best")


class PredictionResponse(BaseModel):
    predictions: list
    probabilities: list[float] | None
    model_name: str
    model_version: int


class HealthResponse(BaseModel):
    status: str
    message: str

class UploadResponse(BaseModel):
    dataset_name: str
    n_rows: int
    n_columns: int
    validation_errors: list[str]
    validation_warnings: list[str]


class TrainRequest(BaseModel):
    dataset_name: str
    target_column: str
    cv_folds: int = Field(5, ge=2, le=10)


class TrainResponse(BaseModel):
    best_model_name: str
    leaderboard: list[dict]
    model_version: int