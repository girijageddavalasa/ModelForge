"""Active-learning pre-annotation and review routes."""

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from app.services import active_learning_service, dataset_service, training_service
from app.services.dataset_service import DatasetNotFoundError
from app.services.training_service import TrainingError
from app.workers import training_worker

active_learning = Blueprint("active_learning", __name__)


@active_learning.route("/datasets/<int:dataset_id>/active-learning", methods=["GET", "POST"])
def dashboard(dataset_id: int) -> str:
    """Queue pre-annotation and display uncertainty-ranked review state."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
    except DatasetNotFoundError:
        abort(404)
    if request.method == "POST":
        try:
            job = active_learning_service.create_preannotation_job(
                dataset, request.form.get("confidence_threshold", "0.25")
            )
            try:
                pid = training_worker.start_training_job(job.id)
            except Exception as error:
                training_service.mark_launch_failed(job.id, "The pre-annotation worker could not start.")
                raise TrainingError("The pre-annotation worker could not start.") from error
            training_service.record_worker_pid(job.id, pid)
        except TrainingError as error:
            flash(str(error), "danger")
        else:
            flash("Pre-annotation job queued.", "success")
            return redirect(url_for("active_learning.dashboard", dataset_id=dataset.id, job_id=job.id))
    job = None
    job_id = request.args.get("job_id", type=int)
    if job_id:
        try:
            candidate = training_service.get_training_job(job_id)
            if candidate.dataset_version.dataset_id == dataset.id and candidate.configuration_json.get("workflow") == "preannotation":
                job = candidate
        except LookupError:
            pass
    return render_template(
        "active_learning/dashboard.html",
        dataset=dataset,
        queue=active_learning_service.review_queue(dataset),
        job=job,
    )


@active_learning.post("/datasets/<int:dataset_id>/active-learning/finalize")
def finalize(dataset_id: int) -> str:
    """Create a reviewed dataset version and hand off to manual retraining."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
        version = active_learning_service.create_reviewed_version(dataset)
    except DatasetNotFoundError:
        abort(404)
    except TrainingError as error:
        flash(str(error), "danger")
        return redirect(url_for("active_learning.dashboard", dataset_id=dataset_id))
    flash(f"Reviewed dataset version v{version.version_number} created. Configure retraining when ready.", "success")
    return redirect(url_for("training.configure_yolo", dataset_id=dataset_id))
