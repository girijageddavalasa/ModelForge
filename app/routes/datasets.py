"""Dataset upload and browsing routes."""

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from app.services import dataset_service, project_service
from app.services.dataset_service import DatasetNotFoundError, DatasetValidationError
from app.services.project_service import ProjectNotFoundError

datasets = Blueprint("datasets", __name__)


@datasets.get("/projects/<int:project_id>/datasets")
def index(project_id: int) -> str:
    """List datasets owned by a project."""
    try:
        project = project_service.get_project(project_id)
    except ProjectNotFoundError:
        abort(404)
    return render_template(
        "datasets/index.html",
        project=project,
        datasets=dataset_service.list_project_datasets(project_id),
    )


@datasets.route("/projects/<int:project_id>/datasets/upload", methods=["GET", "POST"])
def upload(project_id: int) -> str:
    """Display and process the secure dataset upload form."""
    try:
        project = project_service.get_project(project_id)
    except ProjectNotFoundError:
        abort(404)
    if request.method == "POST":
        try:
            dataset = dataset_service.upload_dataset(
                project,
                request.form.get("name", ""),
                request.files.getlist("files"),
            )
        except DatasetValidationError as error:
            flash(str(error), "danger")
        else:
            flash("Dataset uploaded successfully.", "success")
            return redirect(url_for("datasets.detail", dataset_id=dataset.id))
    return render_template("datasets/upload.html", project=project)


@datasets.get("/datasets/<int:dataset_id>")
def detail(dataset_id: int) -> str:
    """Display dataset metadata and its stored file inventory."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
    except DatasetNotFoundError:
        abort(404)
    return render_template("datasets/detail.html", dataset=dataset)
