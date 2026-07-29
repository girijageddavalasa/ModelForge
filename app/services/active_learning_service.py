"""Active-learning pre-annotation, uncertainty review, and versioning services."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from flask import current_app
from sqlalchemy import delete, func, select

from app.extensions import db
from app.ml import registry
from app.models import Annotation, Dataset, DatasetVersion, TrainingJob
from app.models.entities import utc_now
from app.services.annotation_service import get_classes, image_path
from app.services.model_registry_service import ModelVersionNotFoundError, active_project_model, artifact_path
from app.services.training_service import TrainingError, get_training_job

LOGGER = logging.getLogger(__name__)


def create_preannotation_job(dataset: Dataset, confidence_threshold: float = 0.25) -> TrainingJob:
    """Persist a queued pre-annotation job for the active YOLO model."""
    if dataset.dataset_type != "images":
        raise TrainingError("Pre-annotation requires an image dataset.")
    try:
        threshold = float(confidence_threshold)
    except (TypeError, ValueError) as error:
        raise TrainingError("Confidence threshold must be a number.") from error
    if not 0.01 <= threshold <= 0.99:
        raise TrainingError("Confidence threshold must be between 0.01 and 0.99.")
    try:
        model = active_project_model(dataset.project_id)
    except ModelVersionNotFoundError as error:
        raise TrainingError("Train and activate a YOLO model before pre-annotation.") from error
    if model.model_name != "yolo":
        raise TrainingError("The active model is not a YOLO model.")
    version = _current_version(dataset)
    job = TrainingJob(
        project_id=dataset.project_id,
        dataset_version_id=version.id,
        model_name="yolo_preannotation",
        status="queued",
        progress=0,
        configuration_json={
            "workflow": "preannotation", "model_id": model.id,
            "confidence_threshold": threshold, "results": [],
            "return_url": f"/datasets/{dataset.id}/active-learning",
        },
    )
    db.session.add(job)
    db.session.commit()
    return job


def execute_preannotation_job(job_id: int) -> TrainingJob:
    """Predict unreviewed images and persist model-sourced pending boxes."""
    job = get_training_job(job_id)
    if job.status not in {"queued", "failed"}:
        raise TrainingError(f"Pre-annotation job {job.id} cannot start from status {job.status}.")
    dataset = job.dataset_version.dataset
    version = job.dataset_version
    model_version_id = int(job.configuration_json["model_id"])
    try:
        active = active_project_model(dataset.project_id)
    except ModelVersionNotFoundError as error:
        raise TrainingError("The active YOLO model is unavailable.") from error
    if active.id != model_version_id:
        raise TrainingError("The active model changed before pre-annotation started.")
    classes = get_classes(dataset)
    if not classes:
        raise TrainingError("Configure object classes before pre-annotation.")
    images = _unreviewed_images(version)
    if not images:
        raise TrainingError("No unreviewed images are available for pre-annotation.")
    job.status = "running"
    job.started_at = utc_now()
    job.progress = 5
    job.error_message = None
    db.session.commit()
    try:
        plugin = registry.get("yolo")
        detector = plugin.build_estimator("object_detection", {"base_model": str(artifact_path(active))})
        paths = [image_path(dataset, name) for name in images]
        predictions = detector.predict(
            source=[str(path) for path in paths],
            conf=float(job.configuration_json["confidence_threshold"]),
            verbose=False,
        )
        by_name = {Path(str(result.path)).name: result for result in predictions}
        summaries: dict[str, dict[str, Any]] = {}
        for index, name in enumerate(images, start=1):
            db.session.execute(delete(Annotation).where(
                Annotation.dataset_version_id == version.id,
                Annotation.image_path == name,
                Annotation.source == "model",
                Annotation.status == "pending",
            ))
            result = by_name.get(name)
            confidences: list[float] = []
            count = 0
            if result is not None and getattr(result, "boxes", None) is not None:
                names = getattr(result, "names", {})
                xyxy = _array(result.boxes.xyxy)
                class_ids = _array(result.boxes.cls).astype(int)
                scores = _array(result.boxes.conf)
                for coordinates, class_id, score in zip(xyxy, class_ids, scores):
                    class_name = str(names.get(int(class_id), classes[int(class_id)] if int(class_id) < len(classes) else ""))
                    if class_name not in classes:
                        LOGGER.warning("Skipping predicted class %s not configured in dataset %s", class_name, dataset.id)
                        continue
                    x_min, y_min, x_max, y_max = (float(value) for value in coordinates)
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    confidence = float(score)
                    db.session.add(Annotation(
                        dataset_version_id=version.id, image_path=name, class_name=class_name,
                        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                        source="model", confidence=confidence, status="pending",
                    ))
                    confidences.append(confidence)
                    count += 1
            uncertainty = 1.0 if not confidences else 1.0 - sum(confidences) / len(confidences)
            summaries[name] = {"uncertainty": round(uncertainty, 6), "prediction_count": count, "reviewed": False}
            job.progress = 5 + round(index / len(images) * 90)
            db.session.commit()
        metadata = dict(version.metadata_json)
        metadata["active_learning"] = {
            "model_id": active.id,
            "model_version": active.version_number,
            "confidence_threshold": job.configuration_json["confidence_threshold"],
            "images": summaries,
        }
        version.metadata_json = metadata
        job.configuration_json = {**job.configuration_json, "results": [{"preannotated_images": len(images)}]}
        job.status = "completed"
        job.progress = 100
        job.completed_at = utc_now()
        db.session.commit()
        return job
    except Exception as error:
        job.status = "failed"
        job.error_message = str(error)
        job.completed_at = utc_now()
        db.session.commit()
        LOGGER.exception("Pre-annotation job %s failed", job.id)
        if isinstance(error, TrainingError):
            raise
        raise TrainingError("Pre-annotation failed. Review the application log.") from error


def review_queue(dataset: Dataset) -> list[dict[str, Any]]:
    """Return pre-annotated images in descending uncertainty order."""
    version = _current_version(dataset)
    active_learning = version.metadata_json.get("active_learning", {})
    images = active_learning.get("images", {})
    queue = [{"name": name, **summary} for name, summary in images.items()]
    return sorted(queue, key=lambda item: (-float(item["uncertainty"]), item["name"]))


def mark_reviewed(dataset: Dataset, filename: str) -> None:
    """Mark one image reviewed after its annotations are accepted or corrected."""
    version = _current_version(dataset)
    metadata = dict(version.metadata_json)
    active_learning = dict(metadata.get("active_learning", {}))
    images = dict(active_learning.get("images", {}))
    if filename in images:
        images[filename] = {**images[filename], "reviewed": True}
        active_learning["images"] = images
        metadata["active_learning"] = active_learning
        version.metadata_json = metadata
        db.session.commit()


def create_reviewed_version(dataset: Dataset) -> DatasetVersion:
    """Clone accepted review state into a new immutable dataset version."""
    source = _current_version(dataset)
    queue = review_queue(dataset)
    if not queue:
        raise TrainingError("Run pre-annotation before creating a reviewed version.")
    if any(not item.get("reviewed") for item in queue):
        raise TrainingError("Review every pre-annotated image before creating a new version.")
    next_number = source.version_number + 1
    metadata = {key: value for key, value in source.metadata_json.items() if key != "active_learning"}
    metadata["parent_version_id"] = source.id
    metadata["review_summary"] = {"source_model_id": source.metadata_json["active_learning"]["model_id"], "reviewed_images": len(queue)}
    target = DatasetVersion(
        dataset_id=dataset.id, version_number=next_number,
        record_count=source.record_count, labelled_count=source.labelled_count,
        metadata_json=metadata,
    )
    db.session.add(target)
    db.session.flush()
    accepted = db.session.scalars(select(Annotation).where(
        Annotation.dataset_version_id == source.id,
        Annotation.status == "accepted",
    ))
    for annotation in accepted:
        db.session.add(Annotation(
            dataset_version_id=target.id, image_path=annotation.image_path,
            class_name=annotation.class_name, x_min=annotation.x_min, y_min=annotation.y_min,
            x_max=annotation.x_max, y_max=annotation.y_max, source=annotation.source,
            confidence=annotation.confidence, status="accepted",
        ))
    db.session.commit()
    return target


def _unreviewed_images(version: DatasetVersion) -> list[str]:
    accepted = set(db.session.scalars(select(Annotation.image_path).where(
        Annotation.dataset_version_id == version.id,
        Annotation.status == "accepted",
    ).distinct()))
    return [name for name in version.metadata_json.get("files", []) if name not in accepted]


def _current_version(dataset: Dataset) -> DatasetVersion:
    if dataset.dataset_type != "images" or not dataset.versions:
        raise TrainingError("Active learning requires a versioned image dataset.")
    return max(dataset.versions, key=lambda item: item.version_number)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
