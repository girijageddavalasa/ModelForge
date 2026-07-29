"""YOLO job creation, execution, export, and ONNX validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from flask import current_app

from app.extensions import db
from app.ml import registry
from app.models import Dataset, TrainingJob
from app.models.entities import utc_now
from app.services.model_registry_service import register_job_models
from app.services.training_service import TrainingError, get_training_job
from app.services.yolo_dataset_service import YoloDatasetError, prepare_yolo_dataset

LOGGER = logging.getLogger(__name__)
ALLOWED_BASE_MODELS = {"yolo11n.pt", "yolo11s.pt", "yolo26n.pt"}


def create_yolo_job(dataset: Dataset, config: dict[str, Any]) -> TrainingJob:
    """Validate configuration and persist a queued YOLO training job."""
    validated = validate_config(dataset, config)
    version = max(dataset.versions, key=lambda item: item.version_number)
    job = TrainingJob(
        project_id=dataset.project_id,
        dataset_version_id=version.id,
        model_name="yolo",
        status="queued",
        progress=0,
        configuration_json={"workflow": "yolo", **validated, "results": []},
    )
    db.session.add(job)
    db.session.commit()
    return job


def validate_config(dataset: Dataset, config: dict[str, Any]) -> dict[str, Any]:
    """Normalize safe bounded YOLO settings."""
    if dataset.dataset_type != "images" or dataset.project.task_type != "object_detection":
        raise TrainingError("YOLO training requires an object-detection image dataset.")
    base_model = str(config.get("base_model", "yolo11n.pt"))
    if base_model not in ALLOWED_BASE_MODELS:
        raise TrainingError("Select a supported YOLO base model.")
    try:
        epochs = int(config.get("epochs", 20))
        image_size = int(config.get("image_size", 640))
        batch_size = int(config.get("batch_size", 8))
        confidence = float(config.get("confidence", 0.25))
        random_seed = int(config.get("random_seed", 42))
        validation_fraction = float(config.get("validation_fraction", 0.2))
    except (TypeError, ValueError) as error:
        raise TrainingError("YOLO settings must contain valid numbers.") from error
    if not 1 <= epochs <= 1000:
        raise TrainingError("Epochs must be between 1 and 1000.")
    if image_size not in {320, 416, 512, 640, 768, 1024, 1280}:
        raise TrainingError("Select a supported image size.")
    if not 1 <= batch_size <= 256:
        raise TrainingError("Batch size must be between 1 and 256.")
    if not 0.01 <= confidence <= 0.99:
        raise TrainingError("Confidence must be between 0.01 and 0.99.")
    if not 0.1 <= validation_fraction <= 0.4:
        raise TrainingError("Validation fraction must be between 0.1 and 0.4.")
    device = str(config.get("device", "auto"))
    if device not in {"auto", "cpu", "0"}:
        raise TrainingError("Device must be auto, cpu, or 0.")
    return {"base_model": base_model, "epochs": epochs, "image_size": image_size, "batch_size": batch_size, "confidence": confidence, "random_seed": random_seed, "validation_fraction": validation_fraction, "device": device}


def execute_yolo_job(job_id: int) -> TrainingJob:
    """Prepare data, train YOLO, export ONNX, validate it, and register a version."""
    job = get_training_job(job_id)
    if job.status not in {"queued", "failed"}:
        raise TrainingError(f"Training job {job.id} cannot start from status {job.status}.")
    dataset = job.dataset_version.dataset
    config = validate_config(dataset, job.configuration_json)
    job.status = "running"
    job.progress = 5
    job.started_at = utc_now()
    job.error_message = None
    db.session.commit()
    try:
        prepared = prepare_yolo_dataset(dataset, config["validation_fraction"], config["random_seed"])
        job.progress = 15
        db.session.commit()
        plugin = registry.get("yolo")
        model = plugin.build_estimator("object_detection", config)
        device = _device(config["device"])
        storage_root = Path(current_app.config["STORAGE_PATH"]).resolve()
        run_project = storage_root / "models" / str(dataset.project_id) / f"yolo-job-{job.id}"
        result = model.train(
            data=str(prepared.yaml_path), epochs=config["epochs"], imgsz=config["image_size"],
            batch=config["batch_size"], device=device, seed=config["random_seed"],
            project=str(run_project), name="training", exist_ok=False, verbose=False,
        )
        job.progress = 80
        db.session.commit()
        save_dir = Path(result.save_dir).resolve()
        best_path = save_dir / "weights" / "best.pt"
        last_path = save_dir / "weights" / "last.pt"
        if not best_path.is_file() or not last_path.is_file():
            raise TrainingError("YOLO training did not produce best.pt and last.pt.")
        trained = plugin.build_estimator("object_detection", {"base_model": str(best_path)})
        exported = Path(trained.export(format="onnx", imgsz=config["image_size"], simplify=False, dynamic=False)).resolve()
        if not exported.is_file():
            raise TrainingError("YOLO ONNX export did not produce a file.")
        validation = validate_onnx_export(exported, config["image_size"])
        metrics = _metrics(result)
        job.configuration_json = {
            **job.configuration_json,
            "device_resolved": str(device),
            "yolo_dataset": prepared.root.relative_to(storage_root).as_posix(),
            "last_model_path": last_path.relative_to(storage_root).as_posix(),
            "onnx_validation": validation,
            "results": [{
                "plugin": "yolo", "display_name": "Ultralytics YOLO", "metrics": metrics,
                "artifact_path": best_path.relative_to(storage_root).as_posix(),
                "export_path": exported.relative_to(storage_root).as_posix(),
            }],
        }
        job.status = "completed"
        job.progress = 100
        job.completed_at = utc_now()
        db.session.commit()
        register_job_models(job)
        return job
    except Exception as error:
        job.status = "failed"
        job.error_message = str(error)
        job.completed_at = utc_now()
        db.session.commit()
        LOGGER.exception("YOLO training job %s failed", job.id)
        if isinstance(error, (TrainingError, YoloDatasetError)):
            raise TrainingError(str(error)) from error
        raise TrainingError("YOLO training failed. Review the application log.") from error


def validate_onnx_export(path: Path, image_size: int) -> dict[str, Any]:
    """Check the ONNX graph and execute one zero-valued validation input."""
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    shape = [dimension if isinstance(dimension, int) and dimension > 0 else fallback for dimension, fallback in zip(input_meta.shape, [1, 3, image_size, image_size])]
    outputs = session.run(None, {input_meta.name: np.zeros(shape, dtype=np.float32)})
    if not outputs or not all(np.isfinite(output).all() for output in outputs):
        raise TrainingError("ONNX validation produced invalid output.")
    return {"passed": True, "input_name": input_meta.name, "output_count": len(outputs)}


def _device(requested: str) -> str | int:
    if requested != "auto":
        return 0 if requested == "0" else "cpu"
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def _metrics(result: Any) -> dict[str, float]:
    values = getattr(result, "results_dict", {}) or {}
    metrics: dict[str, float] = {}
    for name, value in values.items():
        try:
            metrics[str(name)] = round(float(value), 6)
        except (TypeError, ValueError):
            continue
    return metrics
