"""Model registry pages, downloads, activation, and prediction API."""

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, send_file, url_for

from app.extensions import csrf
from app.services import model_registry_service, prediction_service
from app.services.model_registry_service import ModelVersionNotFoundError
from app.services.prediction_service import PredictionValidationError

models = Blueprint("models", __name__)


@models.get("/projects/<int:project_id>/models")
def index(project_id: int) -> str:
    """List immutable model versions for a project."""
    try:
        project = model_registry_service.project_for_models(project_id)
    except LookupError:
        abort(404)
    return render_template("models/index.html", project=project, models=model_registry_service.list_project_models(project_id))


@models.get("/models/<int:model_id>")
def detail(model_id: int) -> str:
    """Display metrics, configuration, and prediction testing UI."""
    try:
        model = model_registry_service.get_model_version(model_id)
    except ModelVersionNotFoundError:
        abort(404)
    return render_template("models/detail.html", model=model)


@models.post("/models/<int:model_id>/activate")
def activate(model_id: int) -> str:
    """Mark one immutable version active for its project."""
    try:
        model = model_registry_service.activate_model(model_id)
    except ModelVersionNotFoundError:
        abort(404)
    flash(f"Model v{model.version_number} is now active.", "success")
    return redirect(url_for("models.detail", model_id=model.id))


@models.get("/models/<int:model_id>/download")
def download(model_id: int):
    """Download a registered model artifact without exposing arbitrary paths."""
    try:
        model = model_registry_service.get_model_version(model_id)
        path = model_registry_service.artifact_path(model)
    except ModelVersionNotFoundError:
        abort(404)
    except (ValueError, FileNotFoundError) as error:
        flash(str(error), "danger")
        return redirect(url_for("models.detail", model_id=model_id))
    return send_file(path, as_attachment=True, download_name=f"model-v{model.version_number}-{model.model_name}{path.suffix}")


@models.get("/projects/<int:project_id>/models/compare")
def compare(project_id: int) -> str:
    """Compare selected versions or all project versions."""
    try:
        project = model_registry_service.project_for_models(project_id)
    except LookupError:
        abort(404)
    available = model_registry_service.list_project_models(project_id)
    requested = {int(value) for value in request.args.getlist("model_id") if value.isdigit()}
    selected = [model for model in available if not requested or model.id in requested]
    return render_template("models/compare.html", project=project, models=selected, available=available)


@models.post("/api/models/<int:model_id>/predict/tabular")
@csrf.exempt
def predict_tabular(model_id: int):
    """Run JSON tabular prediction through a registered pipeline."""
    try:
        model = model_registry_service.get_model_version(model_id)
        if not request.is_json:
            raise PredictionValidationError("Content-Type must be application/json.")
        result = prediction_service.predict_tabular(model, request.get_json(silent=True))
    except ModelVersionNotFoundError:
        return jsonify({"error": "Model version not found."}), 404
    except PredictionValidationError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError:
        return jsonify({"error": "Model artifact is unavailable."}), 410
    return jsonify(result)

@models.get("/models/<int:model_id>/export/download")
def download_export(model_id: int):
    """Download a registered ONNX or other export artifact."""
    try:
        model = model_registry_service.get_model_version(model_id)
        path = model_registry_service.export_artifact_path(model)
    except ModelVersionNotFoundError:
        abort(404)
    except (ValueError, FileNotFoundError) as error:
        flash(str(error), "danger")
        return redirect(url_for("models.detail", model_id=model_id))
    return send_file(path, as_attachment=True, download_name=f"model-v{model.version_number}-{model.model_name}{path.suffix}")
