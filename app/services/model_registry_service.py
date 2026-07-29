"""Immutable model-version registry services."""

from __future__ import annotations

from pathlib import Path

from flask import current_app
from sqlalchemy import func, select, update

from app.extensions import db
from app.models import ModelVersion, Project, TrainingJob


class ModelVersionNotFoundError(LookupError):
    """Raised when a model version does not exist."""


def register_job_models(job: TrainingJob) -> list[ModelVersion]:
    """Create immutable versions for every successful candidate result."""
    if job.status != "completed":
        raise ValueError("Only completed jobs can create model versions.")
    existing = list(db.session.scalars(select(ModelVersion).where(ModelVersion.training_job_id == job.id)))
    if existing:
        return existing
    current_max = db.session.scalar(
        select(func.max(ModelVersion.version_number)).where(ModelVersion.project_id == job.project_id)
    ) or 0
    has_active = bool(db.session.scalar(
        select(func.count()).select_from(ModelVersion).where(
            ModelVersion.project_id == job.project_id,
            ModelVersion.is_active.is_(True),
        )
    ))
    versions: list[ModelVersion] = []
    for offset, result in enumerate(job.configuration_json.get("results", []), start=1):
        version = ModelVersion(
            project_id=job.project_id,
            training_job_id=job.id,
            version_number=current_max + offset,
            model_name=result["plugin"],
            model_path=result["artifact_path"],
            export_path=result.get("export_path", result["artifact_path"]),
            metrics_json=result["metrics"],
            configuration_json={
                "display_name": result["display_name"],
                **{
                    key: value
                    for key, value in job.configuration_json.items()
                    if key not in {"results", "worker_pid"}
                },
            },
            is_active=not has_active and offset == 1,
        )
        db.session.add(version)
        versions.append(version)
    db.session.commit()
    return versions


def get_model_version(model_id: int) -> ModelVersion:
    """Return one model version or raise a domain error."""
    model = db.session.get(ModelVersion, model_id)
    if model is None:
        raise ModelVersionNotFoundError(f"Model version {model_id} was not found.")
    return model


def list_project_models(project_id: int) -> list[ModelVersion]:
    """List project model versions newest first."""
    statement = select(ModelVersion).where(ModelVersion.project_id == project_id).order_by(ModelVersion.version_number.desc())
    return list(db.session.scalars(statement))


def activate_model(model_id: int) -> ModelVersion:
    """Atomically mark one project model version active."""
    model = get_model_version(model_id)
    db.session.execute(
        update(ModelVersion).where(ModelVersion.project_id == model.project_id).values(is_active=False)
    )
    model.is_active = True
    db.session.commit()
    return model


def artifact_path(model: ModelVersion) -> Path:
    """Resolve and validate a model artifact path within local storage."""
    root = Path(current_app.config["STORAGE_PATH"]).resolve()
    path = (root / model.model_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Unsafe model artifact path.") from error
    if not path.is_file():
        raise FileNotFoundError("The model artifact is missing.")
    return path


def project_for_models(project_id: int) -> Project:
    """Return a project for model pages."""
    project = db.session.get(Project, project_id)
    if project is None:
        raise LookupError(f"Project {project_id} was not found.")
    return project

def export_artifact_path(model: ModelVersion) -> Path:
    """Resolve and validate a registered export artifact."""
    if not model.export_path:
        raise FileNotFoundError("This model has no export artifact.")
    root = Path(current_app.config["STORAGE_PATH"]).resolve()
    path = (root / model.export_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Unsafe export artifact path.") from error
    if not path.is_file():
        raise FileNotFoundError("The export artifact is missing.")
    return path
