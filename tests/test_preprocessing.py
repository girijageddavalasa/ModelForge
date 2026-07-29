"""Preprocessing approval and pipeline tests."""

from __future__ import annotations

import io

import numpy as np
from flask.testing import FlaskClient

from app.extensions import db
from app.models import Dataset, Project
from app.services.preprocessing_service import build_preprocessor, load_features_and_target


def analyzed_dataset(client: FlaskClient) -> Dataset:
    """Upload and analyze a representative tabular dataset."""
    project = Project(name="Preprocessing", description="", task_type="tabular_classification")
    db.session.add(project)
    db.session.commit()
    content = (
        b"id,age,city,constant,target\n"
        b"1,20,A,x,yes\n2,,B,x,no\n3,40,A,x,no\n4,50,C,x,no\n"
    )
    client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Training", "files": (io.BytesIO(content), "training.csv")},
        content_type="multipart/form-data",
    )
    dataset = db.session.scalar(db.select(Dataset))
    assert dataset is not None
    response = client.post(f"/datasets/{dataset.id}/analysis", data={"target_column": "target"})
    assert response.status_code == 200
    return dataset


def test_approve_decisions_and_build_pipeline(client: FlaskClient) -> None:
    """All decisions persist and produce a fit-ready ColumnTransformer pipeline."""
    dataset = analyzed_dataset(client)
    analysis = dataset.versions[0].metadata_json["analysis"]
    form = {"target_column": "target"}
    for index, item in enumerate(analysis["recommendations"]):
        form[f"decision_{index}"] = "approve" if item["issue"] in {"Constant column", "Identifier-like column"} else "reject"
    response = client.post(f"/datasets/{dataset.id}/preprocessing", data=form, follow_redirects=True)
    assert response.status_code == 200
    assert b"Preprocessing decisions approved" in response.data
    db.session.refresh(dataset)
    config = dataset.versions[0].metadata_json["preprocessing"]
    assert config["status"] == "approved"
    assert "id" in config["excluded_columns"]
    assert "constant" in config["excluded_columns"]
    features, target = load_features_and_target(dataset, config)
    transformed = build_preprocessor(config).fit_transform(features, target)
    assert transformed.shape[0] == 4
    assert np.isfinite(transformed).all()


def test_requires_every_recommendation_decision(client: FlaskClient) -> None:
    """Partial approval submissions are rejected."""
    dataset = analyzed_dataset(client)
    response = client.post(
        f"/datasets/{dataset.id}/preprocessing",
        data={"target_column": "target"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Approve or reject every recommendation" in response.data
    db.session.refresh(dataset)
    assert "preprocessing" not in dataset.versions[0].metadata_json


def test_reject_invalid_preprocessing_target(client: FlaskClient) -> None:
    """The target must be one of the analyzed columns."""
    dataset = analyzed_dataset(client)
    analysis = dataset.versions[0].metadata_json["analysis"]
    form = {"target_column": "unknown"}
    form.update({f"decision_{index}": "reject" for index in range(len(analysis["recommendations"]))})
    response = client.post(f"/datasets/{dataset.id}/preprocessing", data=form)
    assert response.status_code == 200
    assert b"Select a valid target column" in response.data


def test_analysis_required_before_preprocessing(client: FlaskClient) -> None:
    """The workflow redirects unanalyzed datasets to Stage 4."""
    project = Project(name="Raw", description="", task_type="tabular_regression")
    db.session.add(project)
    db.session.commit()
    client.post(
        f"/projects/{project.id}/datasets/upload",
        data={"name": "Raw", "files": (io.BytesIO(b"x,y\n1,2\n"), "raw.csv")},
        content_type="multipart/form-data",
    )
    dataset = db.session.scalar(db.select(Dataset))
    assert dataset is not None
    response = client.get(f"/datasets/{dataset.id}/preprocessing")
    assert response.status_code == 302
    assert response.headers["Location"].endswith(f"/datasets/{dataset.id}/analysis")
