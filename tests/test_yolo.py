"""YOLO dataset conversion, job configuration, and mocked training tests."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

from flask import Flask
from flask.testing import FlaskClient
from PIL import Image

from app.extensions import db
from app.models import Dataset, ModelVersion, Project, TrainingJob
from app.services import yolo_training_service
from app.services.yolo_dataset_service import pixel_box_to_yolo, prepare_yolo_dataset


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(buffer, "PNG")
    return buffer.getvalue()


def labelled_dataset(client: FlaskClient) -> Dataset:
    """Create four labelled images suitable for deterministic splitting."""
    project = Project(name="YOLO", description="", task_type="object_detection")
    db.session.add(project)
    db.session.commit()
    files = [(io.BytesIO(png_bytes()), f"image-{index}.png") for index in range(4)]
    client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Vision", "files": files},
        content_type="multipart/form-data",
    )
    dataset = db.session.scalar(db.select(Dataset))
    assert dataset is not None
    client.post(f"/datasets/{dataset.id}/classes", data={"classes": "person\ncar"})
    for index in range(4):
        response = client.put(
            f"/api/datasets/{dataset.id}/annotations",
            json={
                "image": f"image-{index}.png", "image_width": 100, "image_height": 80,
                "annotations": [{"class_name": "person" if index % 2 else "car", "x_min": 10, "y_min": 8, "x_max": 60, "y_max": 48}],
            },
        )
        assert response.status_code == 200
    return dataset


def test_pixel_box_to_yolo_coordinates() -> None:
    """Pixel corners convert to normalized center, width, and height."""
    assert pixel_box_to_yolo(10, 20, 50, 60, 100, 100) == (0.3, 0.4, 0.4, 0.4)


def test_prepare_standard_yolo_dataset(client: FlaskClient) -> None:
    """Preparation creates deterministic images, labels, and data.yaml."""
    dataset = labelled_dataset(client)
    prepared = prepare_yolo_dataset(dataset, validation_fraction=0.25, random_seed=7)
    assert len(prepared.train_images) == 3
    assert len(prepared.val_images) == 1
    assert prepared.yaml_path.is_file()
    assert (prepared.root / "images" / "train").is_dir()
    assert (prepared.root / "labels" / "val").is_dir()
    label_files = list((prepared.root / "labels").glob("*/*.txt"))
    assert len(label_files) == 4
    values = label_files[0].read_text(encoding="utf-8").split()
    assert values[0] in {"0", "1"}
    assert all(0 <= float(value) <= 1 for value in values[1:])
    repeated = prepare_yolo_dataset(dataset, validation_fraction=0.25, random_seed=7)
    assert repeated.root == prepared.root
    assert repeated.train_images == prepared.train_images


def test_yolo_configuration_validation(client: FlaskClient) -> None:
    """Unsafe base models and unreasonable values are rejected."""
    dataset = labelled_dataset(client)
    valid = yolo_training_service.validate_config(dataset, {"base_model": "yolo11n.pt", "epochs": 5})
    assert valid["epochs"] == 5
    for config in ({"base_model": "../../model.pt"}, {"epochs": 0}, {"confidence": 2}):
        try:
            yolo_training_service.validate_config(dataset, config)
        except Exception as error:
            assert str(error)
        else:
            raise AssertionError("Invalid YOLO configuration was accepted")


def test_mocked_yolo_training_registers_pt_and_onnx(client: FlaskClient, app: Flask, monkeypatch) -> None:
    """A mocked training run preserves PT files, exports ONNX, and registers a version."""
    dataset = labelled_dataset(client)
    job = yolo_training_service.create_yolo_job(dataset, {"epochs": 2, "image_size": 320, "batch_size": 2, "device": "cpu"})

    class FakeModel:
        def train(self, **kwargs):
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            weights = save_dir / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")
            return SimpleNamespace(save_dir=save_dir, results_dict={"metrics/mAP50(B)": 0.75})
        def export(self, **_kwargs):
            path = Path(self.source).with_suffix(".onnx")
            path.write_bytes(b"onnx")
            return str(path)
        def __init__(self, source: str = ""):
            self.source = source

    class FakePlugin:
        def build_estimator(self, _task, config):
            return FakeModel(str(config.get("base_model", "")))

    monkeypatch.setattr(yolo_training_service.registry, "get", lambda _name: FakePlugin())
    monkeypatch.setattr(yolo_training_service, "validate_onnx_export", lambda path, size: {"passed": True, "output_count": 1})
    completed = yolo_training_service.execute_yolo_job(job.id)
    assert completed.status == "completed" and completed.progress == 100
    result = completed.configuration_json["results"][0]
    assert result["artifact_path"].endswith("best.pt")
    assert result["export_path"].endswith("best.onnx")
    assert completed.configuration_json["onnx_validation"]["passed"] is True
    model = db.session.scalar(db.select(ModelVersion))
    assert model is not None and model.model_path.endswith("best.pt") and model.export_path.endswith("best.onnx")
    assert (Path(app.config["STORAGE_PATH"]) / completed.configuration_json["last_model_path"]).is_file()


def test_yolo_route_queues_worker(client: FlaskClient, monkeypatch) -> None:
    """The YOLO form creates a queued process-backed job."""
    dataset = labelled_dataset(client)
    monkeypatch.setattr("app.workers.training_worker.start_training_job", lambda job_id: 777)
    response = client.post(
        f"/datasets/{dataset.id}/train/yolo",
        data={"base_model": "yolo11n.pt", "epochs": "2", "image_size": "320", "batch_size": "2", "confidence": "0.25", "validation_fraction": "0.25", "random_seed": "42", "device": "cpu"},
        follow_redirects=True,
    )
    assert response.status_code == 200 and b"YOLO training job queued" in response.data
    job = db.session.scalar(db.select(TrainingJob))
    assert job is not None and job.configuration_json["workflow"] == "yolo"
    assert job.configuration_json["worker_pid"] == 777


def test_real_onnx_runtime_validation(tmp_path: Path) -> None:
    """The export validator checks and executes a real local ONNX graph."""
    import onnx
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 4, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])
    graph = helper.make_graph([helper.make_node("Identity", ["images"], ["output"])], "identity", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "identity.onnx"
    onnx.save(model, path)
    validation = yolo_training_service.validate_onnx_export(path, 4)
    assert validation["passed"] is True
    assert validation["output_count"] == 1
