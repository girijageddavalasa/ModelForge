"""Primary web and operational routes."""

from flask import Blueprint, jsonify, render_template

from app.services import project_service, system_service

main = Blueprint("main", __name__)


@main.get("/")
def home() -> str:
    """Render the project dashboard."""
    projects = project_service.list_projects()
    return render_template("home.html", projects=projects)


@main.get("/health/live")
def liveness():
    """Report whether the web process is responsive."""
    return jsonify({"status": "ok"})


@main.get("/health/ready")
def readiness():
    """Report whether required local dependencies are available."""
    ready = system_service.database_is_ready()
    return jsonify({"status": "ok" if ready else "unavailable"}), 200 if ready else 503