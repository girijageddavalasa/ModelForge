"""Queued tabular training orchestration and execution logic."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from flask import current_app
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.extensions import db
from app.ml import registry
from app.models import Dataset, TrainingJob
from app.models.entities import utc_now
from app.services.dataset_service import DatasetValidationError
from app.services.preprocessing_service import build_preprocessor, get_preprocessing, load_features_and_target

LOGGER = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Raised when a tabular training request cannot complete."""


def available_plugins(dataset: Dataset) -> list[Any]:
    """Return plugins compatible with the dataset project's task."""
    return registry.for_task(dataset.project.task_type)


def create_training_job(
    dataset: Dataset,
    plugin_names: list[str],
    random_seed: int = 42,
    test_size: float = 0.2,
) -> TrainingJob:
    """Validate a request and persist a queued job without doing training work."""
    selected = _validate_request(dataset, plugin_names, test_size)
    version = max(dataset.versions, key=lambda item: item.version_number)
    job = TrainingJob(
        project_id=dataset.project_id,
        dataset_version_id=version.id,
        model_name=", ".join(selected),
        status="queued",
        progress=0,
        configuration_json={
            "workflow": "tabular",
            "plugins": selected,
            "random_seed": random_seed,
            "test_size": test_size,
            "results": [],
        },
    )
    db.session.add(job)
    db.session.commit()
    LOGGER.info("Queued training job %s", job.id)
    return job


def execute_training_job(job_id: int) -> TrainingJob:
    """Execute one queued job inside an initialized Flask application context."""
    job = get_training_job(job_id)
    if job.status not in {"queued", "failed"}:
        raise TrainingError(f"Training job {job.id} cannot start from status {job.status}.")
    dataset = job.dataset_version.dataset
    selected = list(job.configuration_json["plugins"])
    random_seed = int(job.configuration_json["random_seed"])
    test_size = float(job.configuration_json["test_size"])
    _validate_request(dataset, selected, test_size)
    job.status = "running"
    job.started_at = utc_now()
    job.completed_at = None
    job.error_message = None
    db.session.commit()

    storage_root = Path(current_app.config["STORAGE_PATH"]).resolve()
    run_root = storage_root / "models" / str(dataset.project_id) / uuid.uuid4().hex
    try:
        preprocessing = get_preprocessing(dataset)
        if not preprocessing:
            raise TrainingError("Approve preprocessing before training models.")
        features, target = load_features_and_target(dataset, preprocessing)
        stratify = _stratify_target(dataset.project.task_type, target)
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_seed, stratify=stratify,
        )
        run_root.mkdir(parents=True, exist_ok=False)
        results: list[dict[str, Any]] = []
        for index, name in enumerate(selected, start=1):
            plugin = registry.get(name)
            estimator = plugin.build_estimator(dataset.project.task_type, {"random_seed": random_seed})
            transformer = build_preprocessor(preprocessing).named_steps["preprocessor"]
            pipeline = Pipeline(steps=[("preprocessor", transformer), ("model", estimator)])
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            metrics = _evaluate(dataset.project.task_type, pipeline, x_test, y_test, predictions)
            artifact = run_root / f"{name}.joblib"
            joblib.dump(pipeline, artifact)
            results.append({
                "plugin": name,
                "display_name": plugin.display_name,
                "metrics": metrics,
                "artifact_path": artifact.relative_to(storage_root).as_posix(),
            })
            job.progress = round(index / len(selected) * 100)
            job.configuration_json = {**job.configuration_json, "results": results}
            db.session.commit()
        job.status = "completed"
        job.progress = 100
        job.completed_at = utc_now()
        db.session.commit()
        from app.services.model_registry_service import register_job_models
        register_job_models(job)
        LOGGER.info("Completed training job %s", job.id)
        return job
    except Exception as error:
        if run_root.exists():
            shutil.rmtree(run_root)
        job.status = "failed"
        job.error_message = str(error)
        job.completed_at = utc_now()
        db.session.commit()
        LOGGER.exception("Training job %s failed", job.id)
        if isinstance(error, TrainingError):
            raise
        if isinstance(error, (ValueError, DatasetValidationError)):
            raise TrainingError(str(error)) from error
        raise TrainingError("Model training failed. Review the application log.") from error


def record_worker_pid(job_id: int, pid: int | None) -> None:
    """Persist the worker process identifier for diagnostics."""
    job = get_training_job(job_id)
    job.configuration_json = {**job.configuration_json, "worker_pid": pid}
    db.session.commit()

def mark_launch_failed(job_id: int, message: str) -> None:
    """Mark a queued job failed when its worker process cannot start."""
    job = get_training_job(job_id)
    job.status = "failed"
    job.error_message = message
    job.completed_at = utc_now()
    db.session.commit()


def get_training_job(job_id: int) -> TrainingJob:
    """Return a training job or raise a clear lookup error."""
    job = db.session.get(TrainingJob, job_id)
    if job is None:
        raise LookupError(f"Training job {job_id} was not found.")
    return job


def job_status(job: TrainingJob) -> dict[str, Any]:
    """Serialize progress state for polling clients."""
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "error_message": job.error_message,
        "result_url": f"/training-jobs/{job.id}",
    }


def _validate_request(dataset: Dataset, plugin_names: list[str], test_size: float) -> list[str]:
    """Validate training prerequisites and candidate compatibility."""
    if dataset.dataset_type != "tabular":
        raise TrainingError("Tabular AutoML requires a tabular dataset.")
    if not get_preprocessing(dataset):
        raise TrainingError("Approve preprocessing before training models.")
    selected = list(dict.fromkeys(plugin_names))
    if not selected:
        raise TrainingError("Select at least one candidate model.")
    compatible = {plugin.name for plugin in available_plugins(dataset)}
    if any(name not in compatible for name in selected):
        raise TrainingError("One or more selected models do not support this task.")
    if not 0.1 <= test_size <= 0.4:
        raise TrainingError("Test size must be between 0.1 and 0.4.")
    return selected


def _stratify_target(task_type: str, target: Any) -> Any | None:
    """Use stratification only when every class can be split safely."""
    if task_type != "tabular_classification":
        return None
    counts = target.value_counts()
    return target if len(counts) > 1 and int(counts.min()) >= 2 else None


def _evaluate(task_type: str, pipeline: Pipeline, features: Any, truth: Any, predictions: Any) -> dict[str, Any]:
    """Calculate task-appropriate JSON-safe evaluation metrics."""
    if task_type == "tabular_classification":
        metrics: dict[str, Any] = {
            "accuracy": round(float(accuracy_score(truth, predictions)), 6),
            "precision": round(float(precision_score(truth, predictions, average="weighted", zero_division=0)), 6),
            "recall": round(float(recall_score(truth, predictions, average="weighted", zero_division=0)), 6),
            "f1": round(float(f1_score(truth, predictions, average="weighted", zero_division=0)), 6),
            "confusion_matrix": confusion_matrix(truth, predictions).tolist(),
        }
        model = pipeline.named_steps["model"]
        if len(np.unique(truth)) == 2 and hasattr(model, "predict_proba"):
            metrics["roc_auc"] = round(float(roc_auc_score(truth, pipeline.predict_proba(features)[:, 1])), 6)
        return metrics
    mse = float(mean_squared_error(truth, predictions))
    return {
        "mae": round(float(mean_absolute_error(truth, predictions)), 6),
        "mse": round(mse, 6),
        "rmse": round(float(np.sqrt(mse)), 6),
        "r2": round(float(r2_score(truth, predictions)), 6),
    }
