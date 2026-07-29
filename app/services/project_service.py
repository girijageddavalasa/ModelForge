"""Business rules for project management."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select

from app.extensions import db
from app.models import Project

TASK_TYPES: dict[str, str] = {
    "tabular_classification": "Tabular classification",
    "tabular_regression": "Tabular regression",
    "object_detection": "Object detection",
}


class ProjectNotFoundError(LookupError):
    """Raised when a requested project does not exist."""


@dataclass(frozen=True)
class ProjectInput:
    """Validated project input."""

    name: str
    description: str
    task_type: str


def validate_project_input(name: str, description: str, task_type: str) -> ProjectInput:
    """Normalize and validate project form values."""
    clean_name = name.strip()
    clean_description = description.strip()
    if not clean_name:
        raise ValueError("Project name is required.")
    if len(clean_name) > 120:
        raise ValueError("Project name must be 120 characters or fewer.")
    if len(clean_description) > 5000:
        raise ValueError("Description must be 5,000 characters or fewer.")
    if task_type not in TASK_TYPES:
        raise ValueError("Select a valid project type.")
    return ProjectInput(clean_name, clean_description, task_type)


def list_projects() -> list[Project]:
    """Return all projects, newest first."""
    statement = select(Project).order_by(Project.created_at.desc())
    return list(db.session.scalars(statement))


def get_project(project_id: int) -> Project:
    """Return one project or raise a domain-specific error."""
    project = db.session.get(Project, project_id)
    if project is None:
        raise ProjectNotFoundError(f"Project {project_id} was not found.")
    return project


def create_project(name: str, description: str, task_type: str) -> Project:
    """Validate and persist a new project."""
    data = validate_project_input(name, description, task_type)
    project = Project(name=data.name, description=data.description, task_type=data.task_type)
    db.session.add(project)
    db.session.commit()
    return project


def update_project(project_id: int, name: str, description: str, task_type: str) -> Project:
    """Validate and update an existing project."""
    project = get_project(project_id)
    data = validate_project_input(name, description, task_type)
    project.name = data.name
    project.description = data.description
    project.task_type = data.task_type
    db.session.commit()
    return project


def delete_project(project_id: int) -> None:
    """Delete a project and its dependent records."""
    project = get_project(project_id)
    db.session.delete(project)
    db.session.commit()
