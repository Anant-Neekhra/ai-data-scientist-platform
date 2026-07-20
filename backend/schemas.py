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