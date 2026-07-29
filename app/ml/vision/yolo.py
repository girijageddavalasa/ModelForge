"""Ultralytics YOLO model plugin."""

from typing import Any

from sklearn.base import BaseEstimator

from app.ml.base_plugin import ModelPlugin


class YoloPlugin(ModelPlugin):
    """Lazy Ultralytics YOLO integration for object detection."""

    name = "yolo"
    display_name = "Ultralytics YOLO"
    supported_tasks = frozenset({"object_detection"})

    def build_estimator(self, task_type: str, config: dict[str, Any]) -> BaseEstimator:
        """Load a YOLO base model only when vision training starts."""
        if task_type != "object_detection":
            raise ValueError(f"Unsupported task type: {task_type}.")
        from ultralytics import YOLO
        return YOLO(str(config.get("base_model", "yolo11n.pt")))  # type: ignore[return-value]
