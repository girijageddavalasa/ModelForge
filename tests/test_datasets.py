"""Dataset upload integration tests."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from flask import Flask
from flask.testing import FlaskClient
from PIL import Image

from app.extensions import db
from app.models import Dataset, Project


def create_project(task_type: str) -> Project:
    """Persist a project for an upload test."""
    project = Project(name="Upload test", description="", task_type=task_type)
    db.session.add(project)
    db.session.commit()
    return project


def image_bytes(image_format: str = "PNG") -> bytes:
    """Create a small valid image payload."""
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="blue").save(buffer, format=image_format)
    return buffer.getvalue()


def test_upload_valid_csv(client: FlaskClient, app: Flask) -> None:
    """A valid CSV is stored and registered with an initial version."""
    project = create_project("tabular_classification")
    response = client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Customers", "files": (io.BytesIO(b"age,target\n20,yes\n"), "customers.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Dataset uploaded successfully" in response.data
    dataset = db.session.scalar(db.select(Dataset))
    assert dataset is not None
    assert dataset.dataset_type == "tabular"
    stored = Path(app.config["STORAGE_PATH"]) / dataset.storage_path / "customers.csv"
    assert stored.read_bytes() == b"age,target\n20,yes\n"


def test_reject_invalid_csv(client: FlaskClient) -> None:
    """A header-only CSV is rejected without creating metadata."""
    project = create_project("tabular_regression")
    response = client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Empty", "files": (io.BytesIO(b"feature,target\n"), "empty.csv")},
        content_type="multipart/form-data",
    )
    assert b"at least one data row" in response.data
    assert db.session.scalar(db.select(db.func.count()).select_from(Dataset)) == 0


def test_upload_multiple_images(client: FlaskClient) -> None:
    """Multiple valid images form an image dataset."""
    project = create_project("object_detection")
    response = client.post(
        f"/projects/{project.id}/datasets/upload",
        data={
            "name": "Road signs",
            "files": [
                (io.BytesIO(image_bytes()), "one.png"),
                (io.BytesIO(image_bytes()), "two.png"),
            ],
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"2" in response.data
    dataset = db.session.scalar(db.select(Dataset))
    assert dataset is not None
    assert dataset.versions[0].record_count == 2


def test_reject_corrupted_image(client: FlaskClient) -> None:
    """An image extension cannot disguise invalid bytes."""
    project = create_project("object_detection")
    response = client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Broken", "files": (io.BytesIO(b"not-image"), "broken.png")},
        content_type="multipart/form-data",
    )
    assert b"Corrupted or invalid image" in response.data


def test_upload_image_zip(client: FlaskClient) -> None:
    """A ZIP containing valid images is safely ingested."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/sign.png", image_bytes())
    archive.seek(0)
    project = create_project("object_detection")
    response = client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Archive", "files": (archive, "images.zip")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"sign.png" in response.data


def test_upload_zip_ignores_macos_metadata(client: FlaskClient) -> None:
    """macOS ZIP metadata is ignored while real images are ingested."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("train/.DS_Store", b"metadata")
        bundle.writestr("__MACOSX/train/._sign.png", b"apple-double")
        bundle.writestr("train/sign.png", image_bytes())
        bundle.writestr("train/road.png", image_bytes())
    archive.seek(0)
    project = create_project("object_detection")
    response = client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Mac archive", "files": (archive, "images.zip")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    dataset = db.session.scalar(db.select(Dataset))
    assert dataset is not None
    assert dataset.versions[0].record_count == 2
    assert dataset.versions[0].metadata_json["files"] == ["sign.png", "road.png"]

def test_reject_zip_path_traversal(client: FlaskClient) -> None:
    """ZIP members cannot escape the dataset directory."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.png", image_bytes())
    archive.seek(0)
    project = create_project("object_detection")
    response = client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Unsafe", "files": (archive, "images.zip")},
        content_type="multipart/form-data",
    )
    assert b"Unsafe path in ZIP archive" in response.data


def test_reject_duplicate_image_names(client: FlaskClient) -> None:
    """Duplicate image basenames are rejected."""
    project = create_project("object_detection")
    response = client.post(
        f"/projects/{project.id}/datasets/upload",
        data={
            "name": "Duplicates",
            "files": [
                (io.BytesIO(image_bytes()), "same.png"),
                (io.BytesIO(image_bytes()), "SAME.PNG"),
            ],
        },
        content_type="multipart/form-data",
    )
    assert b"Duplicate file name" in response.data


def test_reject_oversized_request(client: FlaskClient, app: Flask) -> None:
    """Requests larger than the configured limit receive a clear 413 page."""
    app.config["MAX_CONTENT_LENGTH"] = 100
    project = create_project("tabular_classification")
    response = client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Large", "files": (io.BytesIO(b"a,b\n" + b"1,2\n" * 100), "large.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert b"exceeds the configured size limit" in response.data
