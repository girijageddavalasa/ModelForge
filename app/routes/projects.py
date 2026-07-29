"""Project management web routes."""

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from app.services import project_service
from app.services.project_service import ProjectNotFoundError

projects = Blueprint("projects", __name__, url_prefix="/projects")


@projects.get("")
def index() -> str:
    """List all projects."""
    return render_template("projects/index.html", projects=project_service.list_projects())


@projects.route("/new", methods=["GET", "POST"])
def create() -> str:
    """Display and process the project creation form."""
    if request.method == "POST":
        try:
            project = project_service.create_project(
                request.form.get("name", ""),
                request.form.get("description", ""),
                request.form.get("task_type", ""),
            )
        except ValueError as error:
            flash(str(error), "danger")
        else:
            flash("Project created successfully.", "success")
            return redirect(url_for("projects.detail", project_id=project.id))
    return render_template("projects/form.html", project=None, task_types=project_service.TASK_TYPES)


@projects.get("/<int:project_id>")
def detail(project_id: int) -> str:
    """Display one project and its related metadata summaries."""
    try:
        project = project_service.get_project(project_id)
    except ProjectNotFoundError:
        abort(404)
    return render_template("projects/detail.html", project=project)


@projects.route("/<int:project_id>/edit", methods=["GET", "POST"])
def edit(project_id: int) -> str:
    """Display and process the project edit form."""
    try:
        project = project_service.get_project(project_id)
        if request.method == "POST":
            project = project_service.update_project(
                project_id,
                request.form.get("name", ""),
                request.form.get("description", ""),
                request.form.get("task_type", ""),
            )
            flash("Project updated successfully.", "success")
            return redirect(url_for("projects.detail", project_id=project.id))
    except ProjectNotFoundError:
        abort(404)
    except ValueError as error:
        flash(str(error), "danger")
    return render_template("projects/form.html", project=project, task_types=project_service.TASK_TYPES)


@projects.post("/<int:project_id>/delete")
def delete(project_id: int) -> str:
    """Delete one project."""
    try:
        project_service.delete_project(project_id)
    except ProjectNotFoundError:
        abort(404)
    flash("Project deleted.", "success")
    return redirect(url_for("projects.index"))
