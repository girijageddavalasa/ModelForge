"""Dispatch a queued job to its workflow-specific executor."""

from app.services.training_service import execute_training_job, get_training_job


def execute_job(job_id: int):
    """Execute tabular or YOLO work based on persisted job configuration."""
    job = get_training_job(job_id)
    if job.configuration_json.get("workflow") == "preannotation":
        from app.services.active_learning_service import execute_preannotation_job
        return execute_preannotation_job(job_id)
    if job.configuration_json.get("workflow") == "yolo":
        from app.services.yolo_training_service import execute_yolo_job
        return execute_yolo_job(job_id)
    return execute_training_job(job_id)
