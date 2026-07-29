"""Spawn-safe local process worker for training jobs."""

from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path

from flask import current_app

LOGGER = logging.getLogger(__name__)


def start_training_job(job_id: int) -> int | None:
    """Start a detached local process and return its operating-system PID."""
    database_uri = str(current_app.config["SQLALCHEMY_DATABASE_URI"])
    storage_path = str(Path(current_app.config["STORAGE_PATH"]).resolve())
    log_level = str(current_app.config["LOG_LEVEL"])
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=run_training_job_process,
        args=(job_id, database_uri, storage_path, log_level),
        name=f"modelforge-training-{job_id}",
        daemon=False,
    )
    process.start()
    LOGGER.info("Started training worker pid=%s for job=%s", process.pid, job_id)
    return process.pid


def run_training_job_process(job_id: int, database_uri: str, storage_path: str, log_level: str) -> None:
    """Create an isolated Flask context and execute one queued job."""
    from app import create_app
    from app.workers.job_runner import execute_job

    application = create_app(
        "development",
        {
            "DEBUG": False,
            "SQLALCHEMY_DATABASE_URI": database_uri,
            "STORAGE_PATH": Path(storage_path),
            "LOG_LEVEL": log_level,
        },
    )
    with application.app_context():
        try:
            execute_job(job_id)
        except Exception:
            application.logger.exception("Worker failed while executing training job %s", job_id)
