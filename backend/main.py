"""
FastAPI application entrypoint. Run with:
    uvicorn backend.main:app --reload
"""

from fastapi import FastAPI

from backend.routes import prediction
from backend.routes import prediction, dataset, training
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI Data Scientist Platform API",
    description="Automated EDA, training, and prediction for tabular datasets.",
    version="0.1.0",
)

app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "AI Data Scientist Platform API is running"}