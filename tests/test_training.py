"""Tabular AutoML registry, worker boundary, and training tests."""

from __future__ import annotations

import io
from pathlib import Path

import joblib
from flask import Flask
from flask.testing import FlaskClient

from app.extensions import db
from app.ml import registry
from app.models import Dataset, Project, TrainingJob
from app.services import training_service


def prepare_dataset(client: FlaskClient, task_type: str) -> Dataset:
    """Create, upload, analyze, and approve a training fixture."""
    project = Project(name="AutoML", description="", task_type=task_type)
    db.session.add(project)
    db.session.commit()
    if task_type == "tabular_classification":
        rows = ["age,city,target"] + [f"{20 + i},{'A' if i % 2 else 'B'},{'yes' if i % 2 else 'no'}" for i in range(40)]
    else:
        rows = ["feature,group,target"] + [f"{i},{'A' if i % 2 else 'B'},{i * 2.5 + 1}" for i in range(40)]
    client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Train", "files": (io.BytesIO(("\n".join(rows) + "\n").encode()), "train.csv")},
        content_type="multipart/form-data",
    )
    dataset = db.session.scalar(db.select(Dataset))
    assert dataset is not None
    client.post(f"/datasets/{dataset.id}/analysis", data={"target_column": "target"})
    analysis = dataset.versions[0].metadata_json["analysis"]
    decisions = {f"decision_{index}": "reject" for index in range(len(analysis["recommendations"]))}
    client.post(f"/datasets/{dataset.id}/preprocessing", data={"target_column": "target", **decisions})
    return dataset


def test_registry_contains_three_tabular_plugins() -> None:
    """The built-in registry exposes every Stage 6 plugin."""
    assert {plugin.name for plugin in registry.for_task("tabular_classification")} == {"linear", "random_forest", "gradient_boosting"}


def test_execute_queued_classification_job(client: FlaskClient, app: Flask) -> None:
    """A queued job executes, reports progress, and creates loadable artifacts."""
    dataset = prepare_dataset(client, "tabular_classification")
    job = training_service.create_training_job(dataset, ["linear", "random_forest"], 7, 0.25)
    assert job.status == "queued"
    training_service.execute_training_job(job.id)
    db.session.refresh(job)
    assert job.status == "completed" and job.progress == 100
    for result in job.configuration_json["results"]:
        artifact = Path(app.config["STORAGE_PATH"]) / result["artifact_path"]
        assert artifact.is_file() and hasattr(joblib.load(artifact), "predict")
        assert "accuracy" in result["metrics"]


def test_execute_queued_regression_job(client: FlaskClient) -> None:
    """A regression worker run emits all required metrics."""
    dataset = prepare_dataset(client, "tabular_regression")
    job = training_service.create_training_job(dataset, ["gradient_boosting"], 42, 0.25)
    training_service.execute_training_job(job.id)
    metrics = job.configuration_json["results"][0]["metrics"]
    assert set(metrics) == {"mae", "mse", "rmse", "r2"}


def test_route_queues_worker_and_status_api(client: FlaskClient, monkeypatch) -> None:
    """The HTTP request queues work, launches a process boundary, and returns pollable state."""
    dataset = prepare_dataset(client, "tabular_classification")
    monkeypatch.setattr("app.workers.training_worker.start_training_job", lambda job_id: 12345)
    response = client.post(
        f"/datasets/{dataset.id}/train",
        data={"plugins": "linear", "random_seed": "42", "test_size": "0.2"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Training job queued" in response.data
    job = db.session.scalar(db.select(TrainingJob))
    assert job is not None and job.status == "queued"
    assert job.configuration_json["worker_pid"] == 12345
    status = client.get(f"/api/training-jobs/{job.id}/status")
    assert status.status_code == 200
    assert status.get_json() == {
        "id": job.id,
        "status": "queued",
        "progress": 0,
        "error_message": None,
        "result_url": f"/training-jobs/{job.id}",
    }


def test_worker_launch_failure_marks_job_failed(client: FlaskClient, monkeypatch) -> None:
    """A process-start failure is persisted and shown to the user."""
    dataset = prepare_dataset(client, "tabular_classification")
    def fail(_job_id: int) -> int:
        raise OSError("spawn unavailable")
    monkeypatch.setattr("app.workers.training_worker.start_training_job", fail)
    response = client.post(
        f"/datasets/{dataset.id}/train",
        data={"plugins": "linear", "random_seed": "42", "test_size": "0.2"},
    )
    assert b"local training worker could not start" in response.data
    job = db.session.scalar(db.select(TrainingJob))
    assert job is not None and job.status == "failed"


def test_training_requires_candidate(client: FlaskClient) -> None:
    """The training form rejects an empty candidate selection."""
    dataset = prepare_dataset(client, "tabular_classification")
    response = client.post(f"/datasets/{dataset.id}/train", data={"random_seed": "42", "test_size": "0.2"})
    assert response.status_code == 200
    assert b"Select at least one candidate model" in response.data
    assert db.session.scalar(db.select(db.func.count()).select_from(TrainingJob)) == 0


def test_missing_job_status_returns_json_404(client: FlaskClient) -> None:
    """Unknown status identifiers return a structured error."""
    response = client.get("/api/training-jobs/999/status")
    assert response.status_code == 404
    assert response.get_json() == {"error": "Training job not found."}
