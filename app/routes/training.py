"""Tabular training configuration and result routes."""

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from app.services import dataset_service, training_service
from app.services.dataset_service import DatasetNotFoundError
from app.services.training_service import TrainingError

training = Blueprint("training", __name__)


@training.route("/datasets/<int:dataset_id>/train", methods=["GET", "POST"])
def configure(dataset_id: int) -> str:
    """Configure and synchronously run Stage 6 candidate training."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
    except DatasetNotFoundError:
        abort(404)
    plugins = training_service.available_plugins(dataset)
    if request.method == "POST":
        try:
            seed = int(request.form.get("random_seed", "42"))
            test_size = float(request.form.get("test_size", "0.2"))
            job = training_service.train_candidates(dataset, request.form.getlist("plugins"), seed, test_size)
        except (ValueError, TrainingError) as error:
            flash(str(error), "danger")
        else:
            flash("Candidate training completed.", "success")
            return redirect(url_for("training.result", job_id=job.id))
    return render_template("training/configure.html", dataset=dataset, plugins=plugins)


@training.get("/training-jobs/<int:job_id>")
def result(job_id: int) -> str:
    """Display candidate metrics and artifact paths."""
    try:
        job = training_service.get_training_job(job_id)
    except LookupError:
        abort(404)
    return render_template("training/result.html", job=job)
