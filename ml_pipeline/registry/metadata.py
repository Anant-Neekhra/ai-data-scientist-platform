"""
Metadata describing one saved model version. Stored as a small JSON
sidecar next to the pickled pipeline -- readable without unpickling
the model, which matters once a dataset has many saved versions.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone


@dataclass
class ModelMetadata:
    """Everything worth knowing about a saved model without loading it."""

    dataset_name: str
    model_name: str          # e.g. "catboost", "random_forest"
    problem_type: str        # "classification" or "regression"
    version: int              # monotonically increasing per dataset+model_name
    created_at: str           # ISO 8601 timestamp
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    n_train_rows: int = 0
    notes: str = ""

    @staticmethod
    def create(
        dataset_name: str,
        model_name: str,
        problem_type: str,
        version: int,
        metrics: dict[str, float],
        hyperparameters: dict,
        feature_columns: list[str],
        n_train_rows: int,
        notes: str = "",
    ) -> "ModelMetadata":
        """Factory that stamps the current UTC time -- centralizing this
        avoids every caller needing to remember the exact timestamp format."""
        return ModelMetadata(
            dataset_name=dataset_name,
            model_name=model_name,
            problem_type=problem_type,
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            hyperparameters=hyperparameters,
            feature_columns=feature_columns,
            n_train_rows=n_train_rows,
            notes=notes,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "ModelMetadata":
        return ModelMetadata(**data)