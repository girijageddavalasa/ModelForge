"""Image gallery, class, and bounding-box annotation services."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from flask import current_app
from sqlalchemy import delete, func, select

from app.extensions import db
from app.models import Annotation, Dataset, DatasetVersion
from app.services.dataset_service import DatasetValidationError


class AnnotationValidationError(ValueError):
    """Raised when annotation state is invalid or unsafe."""


def image_inventory(dataset: Dataset, status: str | None = None) -> list[dict[str, Any]]:
    """Return image names and labelled state, optionally filtered."""
    version = _image_version(dataset)
    images = list(version.metadata_json.get("files", []))
    counts = dict(db.session.execute(
        select(Annotation.image_path, func.count(Annotation.id))
        .where(Annotation.dataset_version_id == version.id)
        .group_by(Annotation.image_path)
    ).all())
    inventory = [
        {"name": name, "status": "labelled" if counts.get(name, 0) else "unlabelled", "annotation_count": int(counts.get(name, 0))}
        for name in images
    ]
    if status in {"labelled", "unlabelled"}:
        inventory = [item for item in inventory if item["status"] == status]
    return inventory


def get_classes(dataset: Dataset) -> list[str]:
    """Return configured object class names."""
    return list(_image_version(dataset).metadata_json.get("classes", []))


def save_classes(dataset: Dataset, names: list[str]) -> list[str]:
    """Normalize and persist object classes without deleting used classes."""
    cleaned = [name.strip() for name in names if name.strip()]
    if not cleaned:
        raise AnnotationValidationError("Enter at least one class name.")
    if any(len(name) > 100 for name in cleaned):
        raise AnnotationValidationError("Class names must be 100 characters or fewer.")
    if len({name.casefold() for name in cleaned}) != len(cleaned):
        raise AnnotationValidationError("Class names must be unique.")
    version = _image_version(dataset)
    used = set(db.session.scalars(
        select(Annotation.class_name).where(Annotation.dataset_version_id == version.id).distinct()
    ))
    if not used.issubset(set(cleaned)):
        raise AnnotationValidationError("Classes used by saved annotations cannot be removed.")
    metadata = dict(version.metadata_json)
    metadata["classes"] = cleaned
    version.metadata_json = metadata
    db.session.commit()
    return cleaned


def image_path(dataset: Dataset, filename: str) -> Path:
    """Resolve a whitelisted dataset image within the storage root."""
    version = _image_version(dataset)
    if filename not in version.metadata_json.get("files", []):
        raise FileNotFoundError("Image not found in this dataset.")
    root = Path(current_app.config["STORAGE_PATH"]).resolve()
    path = (root / dataset.storage_path / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AnnotationValidationError("Unsafe image path.") from error
    if not path.is_file():
        raise FileNotFoundError("Stored image file is missing.")
    return path


def list_annotations(dataset: Dataset, filename: str) -> list[dict[str, Any]]:
    """Return JSON-ready boxes for one image."""
    version = _image_version(dataset)
    _require_image(version, filename)
    statement = select(Annotation).where(
        Annotation.dataset_version_id == version.id,
        Annotation.image_path == filename,
    ).order_by(Annotation.id)
    return [_serialize(annotation) for annotation in db.session.scalars(statement)]


def save_annotations(
    dataset: Dataset,
    filename: str,
    image_width: int,
    image_height: int,
    boxes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace one image's annotations after strict coordinate validation."""
    version = _image_version(dataset)
    _require_image(version, filename)
    if image_width <= 0 or image_height <= 0:
        raise AnnotationValidationError("Image dimensions must be positive.")
    limit = int(current_app.config["MAX_ANNOTATIONS_PER_IMAGE"])
    if len(boxes) > limit:
        raise AnnotationValidationError(f"A maximum of {limit} boxes is allowed per image.")
    classes = set(get_classes(dataset))
    if boxes and not classes:
        raise AnnotationValidationError("Create at least one class before saving boxes.")
    validated = [_validate_box(box, classes, image_width, image_height) for box in boxes]
    db.session.execute(delete(Annotation).where(
        Annotation.dataset_version_id == version.id,
        Annotation.image_path == filename,
    ))
    saved: list[Annotation] = []
    for box in validated:
        annotation = Annotation(
            dataset_version_id=version.id,
            image_path=filename,
            class_name=box["class_name"],
            x_min=box["x_min"], y_min=box["y_min"],
            x_max=box["x_max"], y_max=box["y_max"],
            source=box["source"], confidence=box["confidence"],
            status="accepted",
        )
        db.session.add(annotation)
        saved.append(annotation)
    db.session.flush()
    version.labelled_count = db.session.scalar(
        select(func.count(func.distinct(Annotation.image_path))).where(Annotation.dataset_version_id == version.id)
    ) or 0
    db.session.commit()
    return [_serialize(annotation) for annotation in saved]


def adjacent_image(dataset: Dataset, filename: str, offset: int) -> str:
    """Return the previous or next image, clamped to the gallery bounds."""
    images = list(_image_version(dataset).metadata_json.get("files", []))
    if filename not in images:
        raise FileNotFoundError("Image not found in this dataset.")
    index = max(0, min(len(images) - 1, images.index(filename) + offset))
    return images[index]


def _image_version(dataset: Dataset) -> DatasetVersion:
    """Return the current version and enforce image dataset type."""
    if dataset.dataset_type != "images":
        raise DatasetValidationError("Image annotation requires an image dataset.")
    if not dataset.versions:
        raise AnnotationValidationError("The dataset has no version.")
    return max(dataset.versions, key=lambda item: item.version_number)


def _require_image(version: DatasetVersion, filename: str) -> None:
    """Require exact membership in the immutable image inventory."""
    if filename not in version.metadata_json.get("files", []):
        raise FileNotFoundError("Image not found in this dataset.")


def _validate_box(box: dict[str, Any], classes: set[str], width: int, height: int) -> dict[str, Any]:
    """Validate one box and normalize numeric values."""
    if not isinstance(box, dict):
        raise AnnotationValidationError("Every annotation must be an object.")
    class_name = str(box.get("class_name", "")).strip()
    if class_name not in classes:
        raise AnnotationValidationError(f"Unknown annotation class: {class_name or 'empty'}.")
    try:
        x_min, y_min, x_max, y_max = (float(box[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
    except (KeyError, TypeError, ValueError) as error:
        raise AnnotationValidationError("Bounding-box coordinates must be numbers.") from error
    if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        raise AnnotationValidationError("Bounding-box coordinates must be finite.")
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise AnnotationValidationError("A bounding box is outside the image boundaries.")
    if x_max <= x_min or y_max <= y_min:
        raise AnnotationValidationError("Bounding boxes must have positive width and height.")
    source = str(box.get("source", "human"))
    if source not in {"human", "model", "human_corrected", "imported"}:
        raise AnnotationValidationError("Annotation source is invalid.")
    confidence_value = box.get("confidence")
    confidence = None if confidence_value is None else float(confidence_value)
    if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise AnnotationValidationError("Confidence must be between 0 and 1.")
    return {"class_name": class_name, "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max, "source": source, "confidence": confidence}


def _serialize(annotation: Annotation) -> dict[str, Any]:
    """Serialize a persisted annotation."""
    return {
        "id": annotation.id,
        "class_name": annotation.class_name,
        "x_min": annotation.x_min, "y_min": annotation.y_min,
        "x_max": annotation.x_max, "y_max": annotation.y_max,
        "source": annotation.source, "confidence": annotation.confidence,
        "status": annotation.status,
    }
