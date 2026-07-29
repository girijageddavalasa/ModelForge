"""Dataset upload, browsing, and analysis routes."""

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from app.services import csv_analysis_service, dataset_service, preprocessing_service, project_service
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
    return render_template("datasets/index.html", project=project, datasets=dataset_service.list_project_datasets(project_id))


@datasets.route("/projects/<int:project_id>/datasets/upload", methods=["GET", "POST"])
def upload(project_id: int) -> str:
    """Display and process the secure dataset upload form."""
    try:
        project = project_service.get_project(project_id)
    except ProjectNotFoundError:
        abort(404)
    if request.method == "POST":
        try:
            dataset = dataset_service.upload_dataset(project, request.form.get("name", ""), request.files.getlist("files"))
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


@datasets.route("/datasets/<int:dataset_id>/analysis", methods=["GET", "POST"])
def analysis(dataset_id: int) -> str:
    """Run or display persisted CSV quality analysis."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
        if dataset.dataset_type != "tabular":
            raise DatasetValidationError("CSV analysis is available only for tabular datasets.")
        result = csv_analysis_service.get_analysis(dataset)
        if request.method == "POST":
            target = request.form.get("target_column", "").strip() or None
            result = csv_analysis_service.analyze_dataset(dataset, target)
            flash("CSV analysis completed.", "success")
    except DatasetNotFoundError:
        abort(404)
    except DatasetValidationError as error:
        flash(str(error), "danger")
        return redirect(url_for("datasets.detail", dataset_id=dataset_id))
    columns = list(result["data_types"]) if result else []
    if not columns:
        version = max(dataset.versions, key=lambda item: item.version_number)
        columns = version.metadata_json.get("analysis", {}).get("data_types", {}).keys()
    return render_template("datasets/analysis.html", dataset=dataset, analysis=result, columns=columns)
@datasets.route("/datasets/<int:dataset_id>/preprocessing", methods=["GET", "POST"])
def preprocessing(dataset_id: int) -> str:
    """Display and persist preprocessing recommendation decisions."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
        if dataset.dataset_type != "tabular":
            raise DatasetValidationError("Preprocessing is available only for tabular datasets.")
        version = max(dataset.versions, key=lambda item: item.version_number)
        analysis_result = version.metadata_json.get("analysis")
        if not analysis_result:
            flash("Run CSV analysis before approving preprocessing.", "warning")
            return redirect(url_for("datasets.analysis", dataset_id=dataset.id))
        saved = preprocessing_service.get_preprocessing(dataset)
        if request.method == "POST":
            decisions = {
                index: request.form.get(f"decision_{index}", "")
                for index in range(len(analysis_result.get("recommendations", [])))
            }
            saved = preprocessing_service.save_decisions(
                dataset,
                request.form.get("target_column", "").strip(),
                decisions,
            )
            flash("Preprocessing decisions approved.", "success")
    except DatasetNotFoundError:
        abort(404)
    except DatasetValidationError as error:
        flash(str(error), "danger")
    return render_template(
        "datasets/preprocessing.html",
        dataset=dataset,
        analysis=analysis_result,
        preprocessing=saved,
    )
