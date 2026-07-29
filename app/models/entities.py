"""Core relational entities for ModelForge Local."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import CheckConstraint, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.extensions import db


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class TimestampMixin:
    """Add creation and modification timestamps."""

    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)


class Project(TimestampMixin, db.Model):
    """A user-created machine-learning project."""

    __tablename__ = "projects"
    __table_args__ = (
        CheckConstraint(
            "task_type IN ('tabular_classification', 'tabular_regression', 'object_detection')",
            name="ck_projects_task_type",
        ),
        Index("ix_projects_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    task_type: Mapped[str] = mapped_column(String(40), nullable=False)

    datasets: Mapped[list[Dataset]] = relationship(back_populates="project", cascade="all, delete-orphan")
    training_jobs: Mapped[list[TrainingJob]] = relationship(back_populates="project", cascade="all, delete-orphan")
    model_versions: Mapped[list[ModelVersion]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Dataset(db.Model):
    """Metadata for a dataset owned by a project."""

    __tablename__ = "datasets"
    __table_args__ = (Index("ix_datasets_project_id", "project_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_type: Mapped[str] = mapped_column(String(40), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)

    project: Mapped[Project] = relationship(back_populates="datasets")
    versions: Mapped[list[DatasetVersion]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class DatasetVersion(db.Model):
    """An immutable logical version of a dataset."""

    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version_number", name="uq_dataset_versions_number"),
        Index("ix_dataset_versions_dataset_id", "dataset_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    labelled_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(db.JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="versions")
    annotations: Mapped[list[Annotation]] = relationship(back_populates="dataset_version", cascade="all, delete-orphan")
    training_jobs: Mapped[list[TrainingJob]] = relationship(back_populates="dataset_version")


class Annotation(db.Model):
    """A bounding-box annotation associated with a dataset version."""

    __tablename__ = "annotations"
    __table_args__ = (
        CheckConstraint("source IN ('human', 'model', 'human_corrected', 'imported')", name="ck_annotations_source"),
        CheckConstraint("x_max > x_min AND y_max > y_min", name="ck_annotations_positive_box"),
        Index("ix_annotations_dataset_version_id", "dataset_version_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    class_name: Mapped[str] = mapped_column(String(255), nullable=False)
    x_min: Mapped[float] = mapped_column(Float, nullable=False)
    y_min: Mapped[float] = mapped_column(Float, nullable=False)
    x_max: Mapped[float] = mapped_column(Float, nullable=False)
    y_max: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)

    dataset_version: Mapped[DatasetVersion] = relationship(back_populates="annotations")


class TrainingJob(db.Model):
    """A local model-training job."""

    __tablename__ = "training_jobs"
    __table_args__ = (
        CheckConstraint("status IN ('queued', 'running', 'completed', 'failed', 'cancelled')", name="ck_training_jobs_status"),
        CheckConstraint("progress >= 0 AND progress <= 100", name="ck_training_jobs_progress"),
        Index("ix_training_jobs_project_id", "project_id"),
        Index("ix_training_jobs_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id", ondelete="RESTRICT"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="queued", nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    configuration_json: Mapped[dict[str, Any]] = mapped_column(db.JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    started_at: Mapped[datetime | None]
    completed_at: Mapped[datetime | None]

    project: Mapped[Project] = relationship(back_populates="training_jobs")
    dataset_version: Mapped[DatasetVersion] = relationship(back_populates="training_jobs")
    model_versions: Mapped[list[ModelVersion]] = relationship(back_populates="training_job")


class ModelVersion(db.Model):
    """Metadata for an immutable trained model artifact."""

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("project_id", "version_number", name="uq_model_versions_project_number"),
        Index("ix_model_versions_project_id", "project_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    training_job_id: Mapped[int] = mapped_column(ForeignKey("training_jobs.id", ondelete="RESTRICT"), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    export_path: Mapped[str | None] = mapped_column(String(1024))
    metrics_json: Mapped[dict[str, Any]] = mapped_column(db.JSON, default=dict, nullable=False)
    configuration_json: Mapped[dict[str, Any]] = mapped_column(db.JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)

    project: Mapped[Project] = relationship(back_populates="model_versions")
    training_job: Mapped[TrainingJob] = relationship(back_populates="model_versions")
