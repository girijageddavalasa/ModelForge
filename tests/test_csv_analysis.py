"""CSV analysis service and UI tests."""

from __future__ import annotations

import io

from flask.testing import FlaskClient

from app.extensions import db
from app.models import Dataset, Project


def upload_csv(client: FlaskClient, content: bytes, task_type: str = "tabular_classification") -> Dataset:
    """Create a project and upload one CSV fixture."""
    project = Project(name="Analysis", description="", task_type=task_type)
    db.session.add(project)
    db.session.commit()
    response = client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Quality data", "files": (io.BytesIO(content), "quality.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    dataset = db.session.scalar(db.select(Dataset))
    assert dataset is not None
    return dataset


def test_analysis_detects_quality_issues(client: FlaskClient) -> None:
    """Analysis returns and persists structured quality findings."""
    dataset = upload_csv(
        client,
        b"id,income,monthly_income,city,target\n"
        b"1,100,10,A,yes\n"
        b"2,200,20,B,no\n"
        b"3,300,30,C,no\n"
        b"4,400,40,D,no\n"
        b"4,400,40,D,no\n"
        b"5,,500,E,no\n",
    )
    response = client.post(
        f"/datasets/{dataset.id}/analysis",
        data={"target_column": "target"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"CSV analysis completed" in response.data
    assert b"Missing values" in response.data
    assert b"Duplicate rows" in response.data
    assert b"High correlation" in response.data
    db.session.refresh(dataset)
    analysis = dataset.versions[0].metadata_json["analysis"]
    assert analysis["row_count"] == 6
    assert analysis["duplicate_rows"] == 1
    assert analysis["missing_values"]["income"] == 1
    assert analysis["class_balance"]["imbalanced"] is True


def test_analysis_detects_constant_and_outlier(client: FlaskClient) -> None:
    """Constant columns and IQR outliers generate recommendations."""
    dataset = upload_csv(
        client,
        b"feature,constant,target\n1,x,1\n2,x,1\n3,x,0\n4,x,0\n100,x,0\n",
    )
    client.post(f"/datasets/{dataset.id}/analysis", data={"target_column": "target"})
    db.session.refresh(dataset)
    analysis = dataset.versions[0].metadata_json["analysis"]
    assert "constant" in analysis["constant_columns"]
    assert analysis["outliers"]["feature"]["count"] == 1


def test_invalid_target_is_rejected(client: FlaskClient) -> None:
    """A target must reference an existing column."""
    dataset = upload_csv(client, b"feature,target\n1,a\n2,b\n")
    response = client.post(
        f"/datasets/{dataset.id}/analysis",
        data={"target_column": "missing"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"selected target column does not exist" in response.data
    db.session.refresh(dataset)
    assert "analysis" not in dataset.versions[0].metadata_json


def test_image_dataset_cannot_be_analyzed(client: FlaskClient) -> None:
    """CSV analysis is unavailable for image datasets."""
    project = Project(name="Vision", description="", task_type="object_detection")
    db.session.add(project)
    db.session.commit()
    response = client.get(f"/datasets/999/analysis")
    assert response.status_code == 404
