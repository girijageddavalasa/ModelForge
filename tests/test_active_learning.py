"""Active-learning pre-annotation, review, and retraining tests."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from flask.testing import FlaskClient
from PIL import Image

from app.extensions import db
from app.models import Annotation, Dataset, DatasetVersion, ModelVersion, Project, TrainingJob
from app.services import active_learning_service
from app.services.training_service import TrainingError


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(buffer, "PNG")
    return buffer.getvalue()


def _dataset_with_model(client: FlaskClient) -> Dataset:
    project = Project(name="Active", description="", task_type="object_detection")
    db.session.add(project)
    db.session.commit()
    files = [(io.BytesIO(_png()), f"image-{index}.png") for index in range(3)]
    client.post(f"/projects/{project.id}/datasets/upload", data={"name": "Pool", "files": files}, content_type="multipart/form-data")
    dataset = db.session.scalar(db.select(Dataset))
    assert dataset is not None
    client.post(f"/datasets/{dataset.id}/classes", data={"classes": "person\ncar"})
    version = dataset.versions[0]
    job = TrainingJob(project_id=project.id, dataset_version_id=version.id, model_name="yolo", status="completed", progress=100)
    db.session.add(job)
    db.session.flush()
    model_path = Path(dataset.storage_path).parent.parent / "models" / "active.pt"
    absolute = Path(client.application.config["STORAGE_PATH"]) / model_path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(b"weights")
    db.session.add(ModelVersion(project_id=project.id, training_job_id=job.id, version_number=1, model_name="yolo", model_path=model_path.as_posix(), is_active=True))
    db.session.commit()
    return dataset


def _run_predictions(dataset: Dataset, monkeypatch) -> TrainingJob:
    class Detector:
        def predict(self, source, **_kwargs):
            results = []
            for index, path in enumerate(source):
                if index == 1:
                    boxes = SimpleNamespace(xyxy=np.empty((0, 4)), cls=np.array([]), conf=np.array([]))
                else:
                    score = 0.9 if index == 0 else 0.4
                    boxes = SimpleNamespace(xyxy=np.array([[10, 8, 60, 48]]), cls=np.array([0]), conf=np.array([score]))
                results.append(SimpleNamespace(path=path, names={0: "person"}, boxes=boxes))
            return results

    class Plugin:
        def build_estimator(self, _task, _config):
            return Detector()

    monkeypatch.setattr(active_learning_service.registry, "get", lambda _name: Plugin())
    job = active_learning_service.create_preannotation_job(dataset, 0.3)
    return active_learning_service.execute_preannotation_job(job.id)


def test_preannotation_sorts_uncertain_images(client: FlaskClient, monkeypatch) -> None:
    """Predictions persist as pending boxes and the least certain image is first."""
    dataset = _dataset_with_model(client)
    job = _run_predictions(dataset, monkeypatch)
    assert job.status == "completed" and job.progress == 100
    queue = active_learning_service.review_queue(dataset)
    assert [item["name"] for item in queue] == ["image-1.png", "image-2.png", "image-0.png"]
    predictions = list(db.session.scalars(db.select(Annotation)))
    assert len(predictions) == 2
    assert all(item.source == "model" and item.status == "pending" for item in predictions)


def test_review_creates_new_dataset_version(client: FlaskClient, monkeypatch) -> None:
    """All reviewed predictions can be frozen into a new retraining version."""
    dataset = _dataset_with_model(client)
    _run_predictions(dataset, monkeypatch)
    try:
        active_learning_service.create_reviewed_version(dataset)
    except TrainingError:
        pass
    else:
        raise AssertionError("Incomplete review was finalized")
    for item in active_learning_service.review_queue(dataset):
        boxes = active_learning_service.db.session.scalars(db.select(Annotation).where(Annotation.image_path == item["name"])).all()
        payload = [{"class_name": box.class_name, "x_min": box.x_min, "y_min": box.y_min, "x_max": box.x_max, "y_max": box.y_max, "source": "human_corrected", "confidence": None} for box in boxes]
        response = client.put(f"/api/datasets/{dataset.id}/annotations", json={"image": item["name"], "image_width": 100, "image_height": 80, "annotations": payload})
        assert response.status_code == 200
    target = active_learning_service.create_reviewed_version(dataset)
    assert target.version_number == 2
    assert target.metadata_json["review_summary"]["reviewed_images"] == 3
    assert db.session.scalar(db.select(db.func.count(Annotation.id)).where(Annotation.dataset_version_id == target.id)) == 2


def test_active_learning_route_queues_worker(client: FlaskClient, monkeypatch) -> None:
    """The dashboard queues pre-annotation through the process worker."""
    dataset = _dataset_with_model(client)
    monkeypatch.setattr("app.workers.training_worker.start_training_job", lambda _job_id: 912)
    response = client.post(f"/datasets/{dataset.id}/active-learning", data={"confidence_threshold": "0.25"}, follow_redirects=True)
    assert response.status_code == 200 and b"Pre-annotation job queued" in response.data
    job = db.session.scalar(db.select(TrainingJob).where(TrainingJob.model_name == "yolo_preannotation"))
    assert job is not None and job.configuration_json["worker_pid"] == 912


def test_preannotation_requires_active_yolo(client: FlaskClient) -> None:
    """Datasets without an active detector receive a domain validation error."""
    project = Project(name="No model", description="", task_type="object_detection")
    db.session.add(project)
    db.session.commit()
    files = [(io.BytesIO(_png()), "image.png")]
    client.post(f"/projects/{project.id}/datasets/upload", data={"name": "Pool", "files": files}, content_type="multipart/form-data")
    dataset = db.session.scalar(db.select(Dataset).where(Dataset.project_id == project.id))
    assert dataset is not None
    try:
        active_learning_service.create_preannotation_job(dataset)
    except TrainingError as error:
        assert "activate" in str(error)
    else:
        raise AssertionError("Pre-annotation accepted a missing model")