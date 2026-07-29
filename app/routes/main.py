"""Primary web routes."""

from flask import Blueprint, render_template

from app.services import project_service

main = Blueprint("main", __name__)


@main.get("/")
def home() -> str:
    """Render the project dashboard."""
    projects = project_service.list_projects()
    return render_template("home.html", projects=projects)
