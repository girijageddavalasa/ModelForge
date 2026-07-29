"""Tabular training configuration, status, and result routes."""

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, url_for

from app.services import dataset_service, training_service, yolo_training_service
from app.services.dataset_service import DatasetNotFoundError
from app.services.training_service import TrainingError
from app.workers import training_worker

training = Blueprint("training", __name__)


@training.route("/datasets/<int:dataset_id>/train", methods=["GET", "POST"])
def configure(dataset_id: int) -> str:
    """Validate a training request, queue it, and launch a local worker."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
    except DatasetNotFoundError:
        abort(404)
    plugins = training_service.available_plugins(dataset)
    if request.method == "POST":
        try:
            seed = int(request.form.get("random_seed", "42"))
            test_size = float(request.form.get("test_size", "0.2"))
            job = training_service.create_training_job(
                dataset, request.form.getlist("plugins"), seed, test_size,
            )
            try:
                pid = training_worker.start_training_job(job.id)
            except Exception as error:
                training_service.mark_launch_failed(job.id, "The local training worker could not start.")
                raise TrainingError("The local training worker could not start.") from error
            job.configuration_json = {**job.configuration_json, "worker_pid": pid}
            from app.extensions import db
            db.session.commit()
        except (ValueError, TrainingError) as error:
            flash(str(error), "danger")
        else:
            flash("Training job queued.", "success")
            return redirect(url_for("training.result", job_id=job.id))
    return render_template("training/configure.html", dataset=dataset, plugins=plugins)


@training.get("/training-jobs/<int:job_id>")
def result(job_id: int) -> str:
    """Display live progress or completed candidate metrics."""
    try:
        job = training_service.get_training_job(job_id)
    except LookupError:
        abort(404)
    return render_template("training/result.html", job=job)


@training.get("/api/training-jobs/<int:job_id>/status")
def status(job_id: int):
    """Return JSON progress for Vanilla JavaScript polling."""
    try:
        job = training_service.get_training_job(job_id)
    except LookupError:
        return jsonify({"error": "Training job not found."}), 404
    return jsonify(training_service.job_status(job))

@training.route("/datasets/<int:dataset_id>/train/yolo", methods=["GET", "POST"])
def configure_yolo(dataset_id: int) -> str:
    """Validate, queue, and launch an object-detection training job."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
    except DatasetNotFoundError:
        abort(404)
    if request.method == "POST":
        try:
            job = yolo_training_service.create_yolo_job(dataset, request.form.to_dict())
            try:
                pid = training_worker.start_training_job(job.id)
            except Exception as error:
                training_service.mark_launch_failed(job.id, "The local YOLO worker could not start.")
                raise TrainingError("The local YOLO worker could not start.") from error
            training_service.record_worker_pid(job.id, pid)
        except TrainingError as error:
            flash(str(error), "danger")
        else:
            flash("YOLO training job queued.", "success")
            return redirect(url_for("training.result", job_id=job.id))
    return render_template("training/yolo_configure.html", dataset=dataset, base_models=sorted(yolo_training_service.ALLOWED_BASE_MODELS))
