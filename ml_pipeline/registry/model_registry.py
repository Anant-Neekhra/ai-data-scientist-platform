"""
Per-dataset model registry: saves/loads fitted pipelines with joblib,
tracks version history and metadata per dataset, and identifies the
current best model for each dataset. This is a simplified,
self-built version of the same idea MLflow's Model Registry provides
at scale -- we build our own first (Phase 9) before integrating
MLflow itself (Phase 11), so the MLflow concepts will map onto
something you've already implemented by hand.
"""

import re
import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from config.settings import MODELS_DIR
from ml_pipeline.registry.metadata import ModelMetadata
from utils.logger import get_logger

logger = get_logger(__name__)


def _sanitize_dataset_name(name: str) -> str:
    """
    Turn an arbitrary dataset filename into a filesystem-safe folder
    name. Strips a trailing extension via plain string splitting
    (NOT pathlib.Path, which would treat any '/' in the name as a
    real path separator and silently discard everything before it --
    exactly the kind of untrusted input this function exists to
    handle safely). Lowercases, replaces anything that isn't
    alphanumeric/underscore/hyphen with underscores.
    """
    # Strip a trailing extension (last ".something") without
    # interpreting the string as a filesystem path.
    stem = re.sub(r"\.[^.]+$", "", name)
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    return safe.lower()

class ModelRegistry:
    """Saves, loads, and tracks versioned models, organized per dataset."""

    def __init__(self, base_dir: Path = MODELS_DIR) -> None:
        self.base_dir = base_dir

    def _dataset_dir(self, dataset_name: str) -> Path:
        d = self.base_dir / _sanitize_dataset_name(dataset_name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _registry_index_path(self, dataset_name: str) -> Path:
        return self._dataset_dir(dataset_name) / "registry.json"

    def _load_index(self, dataset_name: str) -> dict:
        index_path = self._registry_index_path(dataset_name)
        if not index_path.exists():
            return {"versions": [], "current_best": None}
        return json.loads(index_path.read_text(encoding="utf-8"))

    def _save_index(self, dataset_name: str, index: dict) -> None:
        index_path = self._registry_index_path(dataset_name)
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def _next_version(self, dataset_name: str) -> int:
        index = self._load_index(dataset_name)
        if not index["versions"]:
            return 1
        return max(v["version"] for v in index["versions"]) + 1

    def save_model(
        self,
        pipeline: Pipeline,
        dataset_name: str,
        model_name: str,
        problem_type: str,
        metrics: dict[str, float],
        hyperparameters: dict,
        feature_columns: list[str],
        n_train_rows: int,
        set_as_best: bool = True,
    ) -> ModelMetadata:
        """
        Save a fitted pipeline and its metadata as a new version.

        Args:
            pipeline: a FITTED sklearn Pipeline (preprocessing + model).
            dataset_name: original dataset filename -- used to key the
                per-dataset folder and index.
            model_name: e.g. "catboost", "random_forest".
            problem_type: "classification" or "regression".
            metrics: whatever evaluation metrics are relevant (accuracy,
                f1, rmse, r2, ...) -- stored as-is, not interpreted here.
            hyperparameters: the model's actual hyperparameter values,
                for reproducibility/comparison across versions.
            feature_columns: the RAW column names the pipeline expects
                as input (i.e. X.columns before preprocessing) -- needed
                later so a caller knows what shape of data to pass in.
            n_train_rows: size of the training set, useful context when
                comparing versions later.
            set_as_best: if True, this version becomes the dataset's
                "current best" pointer. Set False when saving an
                experimental version you don't want to promote yet.

        Returns:
            The ModelMetadata that was saved alongside the model.
        """
        version = self._next_version(dataset_name)
        timestamp = ModelMetadata.create(
            dataset_name, model_name, problem_type, version,
            metrics, hyperparameters, feature_columns, n_train_rows,
        ).created_at
        # Filesystem-safe timestamp for the filename (colons aren't
        # valid in Windows filenames).
        file_timestamp = timestamp.replace(":", "").replace("-", "").split(".")[0]

        base_filename = f"{model_name}_v{version}_{file_timestamp}"
        model_path = self._dataset_dir(dataset_name) / f"{base_filename}.joblib"
        meta_path = self._dataset_dir(dataset_name) / f"{base_filename}_meta.json"

        metadata = ModelMetadata.create(
            dataset_name=dataset_name,
            model_name=model_name,
            problem_type=problem_type,
            version=version,
            metrics=metrics,
            hyperparameters=hyperparameters,
            feature_columns=feature_columns,
            n_train_rows=n_train_rows,
        )

        joblib.dump(pipeline, model_path)
        meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

        index = self._load_index(dataset_name)
        index["versions"].append(
            {
                "version": version,
                "model_name": model_name,
                "model_path": str(model_path),
                "meta_path": str(meta_path),
                "created_at": timestamp,
            }
        )
        if set_as_best:
            index["current_best"] = version
        self._save_index(dataset_name, index)

        logger.info(
            f"Saved model version {version} ('{model_name}') for dataset "
            f"'{dataset_name}' -- set_as_best={set_as_best}"
        )
        return metadata

    def load_model(self, dataset_name: str, version: int | None = None) -> Pipeline:
        """
        Load a fitted pipeline back into memory.

        Args:
            dataset_name: which dataset's registry to look in.
            version: specific version to load. If None, loads the
                dataset's current "best" version.

        Returns:
            The fitted Pipeline, ready to call .predict() on raw data.
        """
        index = self._load_index(dataset_name)
        target_version = version if version is not None else index["current_best"]

        if target_version is None:
            raise FileNotFoundError(
                f"No models saved for dataset '{dataset_name}' yet."
            )

        entry = next((v for v in index["versions"] if v["version"] == target_version), None)
        if entry is None:
            raise FileNotFoundError(
                f"Version {target_version} not found for dataset '{dataset_name}'. "
                f"Available versions: {[v['version'] for v in index['versions']]}"
            )

        pipeline = joblib.load(entry["model_path"])
        logger.info(f"Loaded version {target_version} for dataset '{dataset_name}'")
        return pipeline

    def load_metadata(self, dataset_name: str, version: int | None = None) -> ModelMetadata:
        """Load just the metadata for a version, without unpickling the model."""
        index = self._load_index(dataset_name)
        target_version = version if version is not None else index["current_best"]

        entry = next((v for v in index["versions"] if v["version"] == target_version), None)
        if entry is None:
            raise FileNotFoundError(
                f"Version {target_version} not found for dataset '{dataset_name}'."
            )

        data = json.loads(Path(entry["meta_path"]).read_text(encoding="utf-8"))
        return ModelMetadata.from_dict(data)

    def list_versions(self, dataset_name: str) -> list[ModelMetadata]:
        """List metadata for every saved version of a dataset, newest first."""
        index = self._load_index(dataset_name)
        metadatas = [
            ModelMetadata.from_dict(json.loads(Path(v["meta_path"]).read_text(encoding="utf-8")))
            for v in index["versions"]
        ]
        return sorted(metadatas, key=lambda m: m.version, reverse=True)

    def set_best(self, dataset_name: str, version: int) -> None:
        """Explicitly promote a specific version to be the current best."""
        index = self._load_index(dataset_name)
        if not any(v["version"] == version for v in index["versions"]):
            raise FileNotFoundError(f"Version {version} not found for dataset '{dataset_name}'.")
        index["current_best"] = version
        self._save_index(dataset_name, index)
        logger.info(f"Set version {version} as current best for dataset '{dataset_name}'")

    def delete_version(self, dataset_name: str, version: int) -> None:
        """Remove a specific version's files and index entry."""
        index = self._load_index(dataset_name)
        entry = next((v for v in index["versions"] if v["version"] == version), None)
        if entry is None:
            raise FileNotFoundError(f"Version {version} not found for dataset '{dataset_name}'.")

        Path(entry["model_path"]).unlink(missing_ok=True)
        Path(entry["meta_path"]).unlink(missing_ok=True)
        index["versions"] = [v for v in index["versions"] if v["version"] != version]
        if index["current_best"] == version:
            # Fall back to the newest remaining version, if any.
            index["current_best"] = max((v["version"] for v in index["versions"]), default=None)
        self._save_index(dataset_name, index)
        logger.info(f"Deleted version {version} for dataset '{dataset_name}'")