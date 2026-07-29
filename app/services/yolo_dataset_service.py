"""Deterministic YOLO dataset generation and coordinate validation."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from flask import current_app
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select

from app.models import Annotation, Dataset, DatasetVersion
from app.services.annotation_service import get_classes, image_path
from app.services.dataset_service import DatasetValidationError


class YoloDatasetError(ValueError):
    """Raised when accepted annotations cannot form a YOLO dataset."""


@dataclass(frozen=True)
class YoloDataset:
    """Prepared YOLO dataset paths and split metadata."""

    root: Path
    yaml_path: Path
    train_images: tuple[str, ...]
    val_images: tuple[str, ...]
    classes: tuple[str, ...]


def pixel_box_to_yolo(x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert a validated pixel box into normalized YOLO center format."""
    if width <= 0 or height <= 0:
        raise YoloDatasetError("Image dimensions must be positive.")
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise YoloDatasetError("Bounding box lies outside image boundaries.")
    if x_max <= x_min or y_max <= y_min:
        raise YoloDatasetError("Bounding box must have positive area.")
    return (
        ((x_min + x_max) / 2) / width,
        ((y_min + y_max) / 2) / height,
        (x_max - x_min) / width,
        (y_max - y_min) / height,
    )


def prepare_yolo_dataset(dataset: Dataset, validation_fraction: float = 0.2, random_seed: int = 42) -> YoloDataset:
    """Generate an immutable standard YOLO directory from accepted annotations."""
    if dataset.dataset_type != "images":
        raise YoloDatasetError("YOLO preparation requires an image dataset.")
    if not 0.1 <= validation_fraction <= 0.4:
        raise YoloDatasetError("Validation fraction must be between 0.1 and 0.4.")
    version = _current_version(dataset)
    classes = get_classes(dataset)
    if not classes:
        raise YoloDatasetError("Create object classes before preparing YOLO data.")
    annotations = list(db_annotations(version.id))
    grouped: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        if annotation.status != "accepted":
            continue
        if annotation.class_name not in classes:
            raise YoloDatasetError(f"Annotation uses unknown class: {annotation.class_name}.")
        grouped.setdefault(annotation.image_path, []).append(annotation)
    labelled = sorted(grouped)
    if len(labelled) < 2:
        raise YoloDatasetError("At least two labelled images are required for train/validation splitting.")
    if len({Path(name).stem.casefold() for name in labelled}) != len(labelled):
        raise YoloDatasetError("Label filenames would collide in YOLO format.")
    inventory = set(version.metadata_json.get("files", []))
    if any(name not in inventory for name in labelled):
        raise YoloDatasetError("An annotation references an image outside the dataset inventory.")

    signature = hashlib.sha256(json.dumps({
        "version": version.id, "classes": classes, "seed": random_seed,
        "validation_fraction": validation_fraction,
        "annotations": [[a.image_path, a.class_name, a.x_min, a.y_min, a.x_max, a.y_max] for a in annotations],
    }, sort_keys=True).encode()).hexdigest()[:16]
    storage_root = Path(current_app.config["STORAGE_PATH"]).resolve()
    destination = storage_root / "yolo" / str(dataset.id) / signature
    if destination.exists():
        return _describe_existing(destination, classes)
    staging = destination.parent / f".{signature}-{uuid.uuid4().hex}.tmp"
    try:
        for split in ("train", "val"):
            (staging / "images" / split).mkdir(parents=True, exist_ok=True)
            (staging / "labels" / split).mkdir(parents=True, exist_ok=True)
        shuffled = labelled.copy()
        random.Random(random_seed).shuffle(shuffled)
        validation_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * validation_fraction)))
        val_names = tuple(sorted(shuffled[:validation_count]))
        train_names = tuple(sorted(shuffled[validation_count:]))
        split_lookup = {name: "val" if name in val_names else "train" for name in labelled}
        for name in labelled:
            source = image_path(dataset, name)
            try:
                with Image.open(source) as image:
                    image.verify()
                with Image.open(source) as image:
                    width, height = image.size
            except (OSError, UnidentifiedImageError) as error:
                raise YoloDatasetError(f"Corrupted image: {name}.") from error
            split = split_lookup[name]
            shutil.copy2(source, staging / "images" / split / name)
            lines: list[str] = []
            for annotation in grouped[name]:
                class_id = classes.index(annotation.class_name)
                center_x, center_y, box_width, box_height = pixel_box_to_yolo(
                    annotation.x_min, annotation.y_min, annotation.x_max, annotation.y_max, width, height,
                )
                lines.append(f"{class_id} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}")
            (staging / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        yaml_path = staging / "data.yaml"
        yaml_path.write_text(yaml.safe_dump({
            "path": destination.as_posix(), "train": "images/train", "val": "images/val",
            "names": {index: name for index, name in enumerate(classes)},
        }, sort_keys=False), encoding="utf-8")
        staging.rename(destination)
        return YoloDataset(destination, destination / "data.yaml", train_names, val_names, tuple(classes))
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def db_annotations(version_id: int):
    """Return annotations in deterministic order."""
    from app.extensions import db
    return db.session.scalars(select(Annotation).where(Annotation.dataset_version_id == version_id).order_by(Annotation.image_path, Annotation.id))


def _current_version(dataset: Dataset) -> DatasetVersion:
    if not dataset.versions:
        raise YoloDatasetError("Dataset has no version.")
    return max(dataset.versions, key=lambda item: item.version_number)


def _describe_existing(root: Path, classes: list[str]) -> YoloDataset:
    train = tuple(sorted(path.name for path in (root / "images" / "train").iterdir() if path.is_file()))
    val = tuple(sorted(path.name for path in (root / "images" / "val").iterdir() if path.is_file()))
    return YoloDataset(root, root / "data.yaml", train, val, tuple(classes))
