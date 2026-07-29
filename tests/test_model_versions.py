"""Model registry, comparison, download, and prediction tests."""

from __future__ import annotations

from pathlib import Path

from flask import Flask
from flask.testing import FlaskClient

from app.extensions import db
from app.models import ModelVersion
from app.services import model_registry_service, training_service
from tests.test_training import prepare_dataset


def trained_versions(client: FlaskClient) -> list[ModelVersion]:
    """Train two candidates and return their immutable versions."""
    dataset = prepare_dataset(client, "tabular_classification")
    job = training_service.create_training_job(dataset, ["linear", "random_forest"], 11, 0.25)
    training_service.execute_training_job(job.id)
    return list(db.session.scalars(db.select(ModelVersion).order_by(ModelVersion.version_number)))


def test_completed_job_creates_immutable_versions(client: FlaskClient, app: Flask) -> None:
    """Every candidate becomes a numbered model version with an artifact."""
    versions = trained_versions(client)
    assert [model.version_number for model in versions] == [1, 2]
    assert versions[0].is_active is True
    assert versions[1].is_active is False
    assert versions[0].training_job_id == versions[1].training_job_id
    for model in versions:
        assert (Path(app.config["STORAGE_PATH"]) / model.model_path).is_file()
        assert model.metrics_json


def test_activate_model_is_exclusive(client: FlaskClient) -> None:
    """Activating a version deactivates the previous project version."""
    first, second = trained_versions(client)
    response = client.post(f"/models/{second.id}/activate", follow_redirects=True)
    assert response.status_code == 200
    assert b"now active" in response.data
    db.session.refresh(first)
    db.session.refresh(second)
    assert first.is_active is False and second.is_active is True


def test_prediction_api_for_single_and_batch_records(client: FlaskClient) -> None:
    """A registered pipeline predicts one object or a batch of objects."""
    model = trained_versions(client)[0]
    response = client.post(
        f"/api/models/{model.id}/predict/tabular",
        json={"age": 31, "city": "A"},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_version"] == 1
    assert len(body["results"]) == 1
    assert "prediction" in body["results"][0]
    assert "probabilities" in body["results"][0]
    batch = client.post(
        f"/api/models/{model.id}/predict/tabular",
        json=[{"age": 28, "city": "B"}, {"age": 45, "city": "new"}],
    )
    assert batch.status_code == 200
    assert len(batch.get_json()["results"]) == 2


def test_prediction_api_validates_json(client: FlaskClient) -> None:
    """Prediction errors are returned as structured JSON."""
    model = trained_versions(client)[0]
    response = client.post(f"/api/models/{model.id}/predict/tabular", data="not-json")
    assert response.status_code == 400
    assert "error" in response.get_json()
    missing = client.post("/api/models/999/predict/tabular", json={"age": 1})
    assert missing.status_code == 404


def test_model_download_and_comparison_pages(client: FlaskClient) -> None:
    """Artifacts download safely and versions render side by side."""
    first, second = trained_versions(client)
    download = client.get(f"/models/{first.id}/download")
    assert download.status_code == 200
    assert "attachment" in download.headers["Content-Disposition"]
    compare = client.get(
        f"/projects/{first.project_id}/models/compare?model_id={first.id}&model_id={second.id}"
    )
    assert compare.status_code == 200
    assert b"Compare model versions" in compare.data
    assert b"v1" in compare.data and b"v2" in compare.data


def test_registration_is_idempotent(client: FlaskClient) -> None:
    """A worker retry cannot duplicate versions for the same completed job."""
    versions = trained_versions(client)
    job = versions[0].training_job
    repeated = model_registry_service.register_job_models(job)
    assert len(repeated) == 2
    assert db.session.scalar(db.select(db.func.count()).select_from(ModelVersion)) == 2
