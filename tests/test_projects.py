"""Project management integration tests."""

from flask.testing import FlaskClient

from app.extensions import db
from app.models import Project


def test_dashboard_empty_state(client: FlaskClient) -> None:
    """The dashboard explains how to create the first project."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"No projects yet" in response.data


def test_create_and_view_project(client: FlaskClient) -> None:
    """A valid project can be created and displayed."""
    response = client.post(
        "/projects/new",
        data={
            "name": "Fraud detection",
            "description": "Classify suspicious transactions.",
            "task_type": "tabular_classification",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Project created successfully" in response.data
    assert b"Fraud detection" in response.data
    assert db.session.scalar(db.select(Project).where(Project.name == "Fraud detection"))


def test_invalid_project_is_rejected(client: FlaskClient) -> None:
    """Invalid form input does not create a project."""
    response = client.post(
        "/projects/new",
        data={"name": "", "description": "", "task_type": "invalid"},
    )
    assert response.status_code == 200
    assert b"Project name is required" in response.data
    assert db.session.scalar(db.select(db.func.count()).select_from(Project)) == 0


def test_edit_project(client: FlaskClient) -> None:
    """An existing project can be updated."""
    client.post(
        "/projects/new",
        data={"name": "Initial", "description": "", "task_type": "tabular_regression"},
    )
    project = db.session.scalar(db.select(Project))
    assert project is not None
    response = client.post(
        f"/projects/{project.id}/edit",
        data={"name": "Updated", "description": "Changed", "task_type": "object_detection"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Project updated successfully" in response.data
    db.session.refresh(project)
    assert project.name == "Updated"
    assert project.task_type == "object_detection"


def test_delete_project(client: FlaskClient) -> None:
    """An existing project can be deleted."""
    client.post(
        "/projects/new",
        data={"name": "Temporary", "description": "", "task_type": "tabular_classification"},
    )
    project = db.session.scalar(db.select(Project))
    assert project is not None
    response = client.post(f"/projects/{project.id}/delete", follow_redirects=True)
    assert response.status_code == 200
    assert b"Project deleted" in response.data
    assert db.session.get(Project, project.id) is None


def test_missing_project_returns_404(client: FlaskClient) -> None:
    """Unknown project identifiers use the custom 404 handler."""
    response = client.get("/projects/999")
    assert response.status_code == 404
    assert b"Page not found" in response.data
