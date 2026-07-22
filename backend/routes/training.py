"""Training endpoint: runs the full comparison + fits the best model."""

from fastapi import APIRouter, HTTPException

from ml_pipeline.data.loader import DatasetLoader, DatasetLoadError
from ml_pipeline.data.splitter import split_dataset
from ml_pipeline.data.schema_detector import SchemaDetector
from ml_pipeline.training.trainer import ModelTrainer
from ml_pipeline.training.comparison import build_leaderboard
from ml_pipeline.registry.model_registry import ModelRegistry
from ml_pipeline.tracking.mlflow_tracker import MLflowTracker
from config.settings import DATASETS_DIR
from backend.schemas import TrainRequest, TrainResponse
from ml_pipeline.data.schema_detector import FeatureType
from pydantic import BaseModel, Field
import requests
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
_registry = ModelRegistry()

class TrainRequest(BaseModel):
    dataset_name: str
    target_column: str
    cv_folds: int = Field(5, ge=2, le=10)
    schema_override: dict[str, str] | None = None

@router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    dataset_path = DATASETS_DIR / request.dataset_name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_name}' not found. Upload it first.")

    try:
        df = DatasetLoader.load_csv(dataset_path)
    except DatasetLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if request.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{request.target_column}' not in dataset columns: {list(df.columns)}",
        )

    df = df.dropna(subset=[request.target_column])
    problem_type = SchemaDetector().detect_problem_type(df, request.target_column)
    X_train, X_test, y_train, y_test = split_dataset(df, target_col=request.target_column)

    tracker = MLflowTracker()
    trainer = ModelTrainer(cv_folds=request.cv_folds)
    results = trainer.compare_models(
        X_train, y_train, problem_type, tracker=tracker,
        dataset_name=request.dataset_name, schema_override=schema_override,
    )
    if not results:
        raise HTTPException(status_code=500, detail="All candidate models failed to train.")

    best_name, fitted_pipeline = trainer.fit_best_model(
        X_train, y_train, results, problem_type, schema_override=schema_override
    )
    leaderboard = build_leaderboard(results, problem_type)

    test_score = fitted_pipeline.score(X_test, y_test)
    metadata = _registry.save_model(
        fitted_pipeline,
        dataset_name=request.dataset_name,
        model_name=best_name,
        problem_type=problem_type.value,
        metrics={"test_score": test_score},
        hyperparameters=fitted_pipeline.named_steps["model"].get_params(),
        feature_columns=list(X_train.columns),
        n_train_rows=len(X_train),
    )

    schema_override = None
    if request.schema_override:
        schema_override = {col: FeatureType(val) for col, val in request.schema_override.items()}

    return TrainResponse(
        best_model_name=best_name,
        leaderboard=leaderboard.to_dict(orient="records"),
        model_version=metadata.version,
    )