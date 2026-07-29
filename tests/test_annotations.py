"""Image gallery and annotation workflow tests."""

from __future__ import annotations

import io

from flask.testing import FlaskClient
from PIL import Image

from app.extensions import db
from app.models import Annotation, Dataset, Project


def png_bytes() -> bytes:
    """Return a small valid test image."""
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def image_dataset(client: FlaskClient) -> Dataset:
    """Create and upload a two-image object-detection dataset."""
    project = Project(name="Annotation", description="", task_type="object_detection")
    db.session.add(project)
    db.session.commit()
    response = client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Images", "files": [(io.BytesIO(png_bytes()), "one.png"), (io.BytesIO(png_bytes()), "two.png")]},
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    dataset = db.session.scalar(db.select(Dataset))
    assert dataset is not None
    return dataset


def test_class_management_and_gallery(client: FlaskClient) -> None:
    """Classes persist and the gallery renders the uploaded images."""
    dataset = image_dataset(client)
    response = client.post(
        f"/datasets/{dataset.id}/classes",
        data={"classes": "person, car"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Object classes saved" in response.data
    db.session.refresh(dataset)
    assert dataset.versions[0].metadata_json["classes"] == ["person", "car"]
    assert b"one.png" in response.data and b"two.png" in response.data


def test_save_and_load_annotations(client: FlaskClient) -> None:
    """Validated boxes persist and update labelled-image count."""
    dataset = image_dataset(client)
    client.post(f"/datasets/{dataset.id}/classes", data={"classes": "person\ncar"})
    payload = {
        "image": "one.png", "image_width": 100, "image_height": 80,
        "annotations": [
            {"class_name": "person", "x_min": 10, "y_min": 5, "x_max": 50, "y_max": 60, "source": "human", "confidence": None},
            {"class_name": "car", "x_min": 55, "y_min": 20, "x_max": 95, "y_max": 70, "source": "human", "confidence": 0.8},
        ],
    }
    response = client.put(f"/api/datasets/{dataset.id}/annotations", json=payload)
    assert response.status_code == 200
    assert response.get_json()["saved"] == 2
    assert db.session.scalar(db.select(db.func.count()).select_from(Annotation)) == 2
    db.session.refresh(dataset)
    assert dataset.versions[0].labelled_count == 1
    loaded = client.get(f"/api/datasets/{dataset.id}/annotations?image=one.png")
    assert loaded.status_code == 200
    assert loaded.get_json()["annotations"][1]["confidence"] == 0.8


def test_replace_annotations_and_filter_gallery(client: FlaskClient) -> None:
    """Saving replaces old boxes and labelled filters use persisted state."""
    dataset = image_dataset(client)
    client.post(f"/datasets/{dataset.id}/classes", data={"classes": "object"})
    url = f"/api/datasets/{dataset.id}/annotations"
    payload = {"image": "one.png", "image_width": 100, "image_height": 80, "annotations": [{"class_name": "object", "x_min": 1, "y_min": 2, "x_max": 20, "y_max": 30}]}
    client.put(url, json=payload)
    payload["annotations"] = []
    client.put(url, json=payload)
    assert db.session.scalar(db.select(db.func.count()).select_from(Annotation)) == 0
    labelled = client.get(f"/datasets/{dataset.id}/images?status=labelled")
    assert b"No images match this filter" in labelled.data
    unlabelled = client.get(f"/datasets/{dataset.id}/images?status=unlabelled")
    assert b"one.png" in unlabelled.data and b"two.png" in unlabelled.data


def test_annotation_validation_rejects_bad_boxes(client: FlaskClient) -> None:
    """Unknown classes, zero-area boxes, and out-of-bounds boxes are rejected."""
    dataset = image_dataset(client)
    client.post(f"/datasets/{dataset.id}/classes", data={"classes": "person"})
    base = {"image": "one.png", "image_width": 100, "image_height": 80}
    for box, message in [
        ({"class_name": "unknown", "x_min": 1, "y_min": 1, "x_max": 10, "y_max": 10}, "Unknown annotation class"),
        ({"class_name": "person", "x_min": 10, "y_min": 1, "x_max": 10, "y_max": 10}, "positive width"),
        ({"class_name": "person", "x_min": 1, "y_min": 1, "x_max": 101, "y_max": 10}, "outside the image boundaries"),
    ]:
        response = client.put(f"/api/datasets/{dataset.id}/annotations", json={**base, "annotations": [box]})
        assert response.status_code == 400
        assert message in response.get_json()["error"]
    assert db.session.scalar(db.select(db.func.count()).select_from(Annotation)) == 0


def test_image_serving_and_editor_are_whitelisted(client: FlaskClient) -> None:
    """Only inventory images can be served or opened in the editor."""
    dataset = image_dataset(client)
    client.post(f"/datasets/{dataset.id}/classes", data={"classes": "object"})
    image = client.get(f"/datasets/{dataset.id}/images/one.png")
    assert image.status_code == 200 and image.mimetype == "image/png"
    assert client.get(f"/datasets/{dataset.id}/images/missing.png").status_code == 404
    editor = client.get(f"/datasets/{dataset.id}/annotate/one.png")
    assert editor.status_code == 200
    assert b"annotation_canvas.js" in editor.data and b"konva" in editor.data.lower()


def test_used_class_cannot_be_removed(client: FlaskClient) -> None:
    """Class updates preserve referential meaning of existing boxes."""
    dataset = image_dataset(client)
    client.post(f"/datasets/{dataset.id}/classes", data={"classes": "person\ncar"})
    client.put(
        f"/api/datasets/{dataset.id}/annotations",
        json={"image": "one.png", "image_width": 100, "image_height": 80, "annotations": [{"class_name": "person", "x_min": 1, "y_min": 1, "x_max": 10, "y_max": 10}]},
    )
    response = client.post(f"/datasets/{dataset.id}/classes", data={"classes": "car"}, follow_redirects=True)
    assert b"used by saved annotations cannot be removed" in response.data
