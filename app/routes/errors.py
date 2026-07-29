"""Application error handlers."""

from flask import Blueprint, current_app, render_template

errors = Blueprint("errors", __name__)


@errors.app_errorhandler(404)
def not_found(_error: Exception) -> tuple[str, int]:
    """Render the not-found page."""
    return render_template("errors/404.html"), 404


@errors.app_errorhandler(500)
def internal_server_error(error: Exception) -> tuple[str, int]:
    """Log an unhandled error and render a safe response."""
    current_app.logger.error(
        "Unhandled server error",
        exc_info=(type(error), error, error.__traceback__),
    )
    return render_template("errors/500.html"), 500
