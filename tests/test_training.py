"""Tabular AutoML plugin and training tests."""

from __future__ import annotations

import io
from pathlib import Path

import joblib
from flask import Flask
from flask.testing import FlaskClient

from app.extensions import db
from app.ml import registry
from app.models import Dataset, Project, TrainingJob


def prepare_dataset(client: FlaskClient, task_type: str) -> Dataset:
    """Create, upload, analyze, and approve a training fixture."""
    project = Project(name="AutoML", description="", task_type=task_type)
    db.session.add(project)
    db.session.commit()
    if task_type == "tabular_classification":
        rows = ["age,city,target"] + [f"{20 + index},{'A' if index % 2 else 'B'},{'yes' if index % 2 else 'no'}" for index in range(40)]
        target = "target"
    else:
        rows = ["feature,group,target"] + [f"{index},{'A' if index % 2 else 'B'},{index * 2.5 + 1}" for index in range(40)]
        target = "target"
    client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Train", "files": (io.BytesIO(("\n".join(rows) + "\n").encode()), "train.csv")},
        content_type="multipart/form-data",
    )
    dataset = db.session.scalar(db.select(Dataset))
    assert dataset is not None
    client.post(f"/datasets/{dataset.id}/analysis", data={"target_column": target})
    analysis = dataset.versions[0].metadata_json["analysis"]
    decisions = {f"decision_{index}": "reject" for index in range(len(analysis["recommendations"]))}
    client.post(f"/datasets/{dataset.id}/preprocessing", data={"target_column": target, **decisions})
    return dataset


def test_registry_contains_three_tabular_plugins() -> None:
    """The built-in registry exposes every Stage 6 plugin."""
    names = {plugin.name for plugin in registry.for_task("tabular_classification")}
    assert names == {"linear", "random_forest", "gradient_boosting"}
    assert registry.get("random_forest").display_name == "Random Forest"


def test_train_classification_candidates(client: FlaskClient, app: Flask) -> None:
    """Classification candidates produce metrics and loadable joblib pipelines."""
    dataset = prepare_dataset(client, "tabular_classification")
    response = client.post(
        f"/datasets/{dataset.id}/train",
        data={"plugins": ["linear", "random_forest"], "random_seed": "7", "test_size": "0.25"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Candidate training completed" in response.data
    job = db.session.scalar(db.select(TrainingJob))
    assert job is not None and job.status == "completed" and job.progress == 100
    assert len(job.configuration_json["results"]) == 2
    for result in job.configuration_json["results"]:
        assert "accuracy" in result["metrics"]
        artifact = Path(app.config["STORAGE_PATH"]) / result["artifact_path"]
        assert artifact.is_file()
        assert hasattr(joblib.load(artifact), "predict")


def test_train_regression_candidate(client: FlaskClient) -> None:
    """Regression training emits MAE, MSE, RMSE, and R-squared."""
    dataset = prepare_dataset(client, "tabular_regression")
    response = client.post(
        f"/datasets/{dataset.id}/train",
        data={"plugins": "gradient_boosting", "random_seed": "42", "test_size": "0.25"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    job = db.session.scalar(db.select(TrainingJob))
    metrics = job.configuration_json["results"][0]["metrics"]
    assert set(metrics) == {"mae", "mse", "rmse", "r2"}


def test_training_requires_candidate(client: FlaskClient) -> None:
    """The training form rejects an empty candidate selection."""
    dataset = prepare_dataset(client, "tabular_classification")
    response = client.post(
        f"/datasets/{dataset.id}/train",
        data={"random_seed": "42", "test_size": "0.2"},
    )
    assert response.status_code == 200
    assert b"Select at least one candidate model" in response.data
    assert db.session.scalar(db.select(db.func.count()).select_from(TrainingJob)) == 0
