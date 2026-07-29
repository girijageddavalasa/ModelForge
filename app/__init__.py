"""Application factory for ModelForge Local."""

from __future__ import annotations

from logging.config import dictConfig
from pathlib import Path

from flask import Flask

from app.config import CONFIGURATIONS, Config
from app.extensions import bootstrap, csrf, db, migrate
from app.routes.active_learning import active_learning
from app.routes.annotations import annotations
from app.routes.datasets import datasets
from app.routes.errors import errors
from app.routes.main import main
from app.routes.models import models
from app.routes.projects import projects
from app.routes.training import training
from app.security import configure_request_controls


def create_app(config_name: str | None = None, config_overrides: dict[str, object] | None = None) -> Flask:
    """Create and configure a ModelForge Local Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    selected_config = config_name or Config.environment_name()
    config_class = CONFIGURATIONS.get(selected_config)
    if config_class is None:
        valid_names = ", ".join(sorted(CONFIGURATIONS))
        raise ValueError(f"Unknown configuration {selected_config!r}. Choose one of: {valid_names}.")
    app.config.from_object(config_class)
    if config_overrides:
        app.config.update(config_overrides)
    if selected_config == "production" and (
        app.config["SECRET_KEY"] == "development-only-secret-key" or len(str(app.config["SECRET_KEY"])) < 32
    ):
        raise RuntimeError("SECRET_KEY must contain at least 32 characters in production.")

    create_runtime_directories(app)
    configure_logging(app)
    initialize_extensions(app)
    import_models()
    register_routes(app)
    register_error_handlers(app)
    configure_request_controls(app)
    app.logger.info("ModelForge Local initialized with %s configuration", selected_config)
    return app


def configure_logging(app: Flask) -> None:
    """Configure console and size-rotated local file logging."""
    handlers: dict[str, dict[str, object]] = {
        "console": {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stdout"}
    }
    root_handlers = ["console"]
    if app.config.get("LOG_PATH"):
        log_path = Path(app.config["LOG_PATH"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler", "formatter": "default",
            "filename": str(log_path), "maxBytes": app.config["LOG_MAX_BYTES"],
            "backupCount": app.config["LOG_BACKUP_COUNT"], "encoding": "utf-8",
        }
        root_handlers.append("file")
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": "%(asctime)s %(levelname)s [%(name)s] %(message)s"}},
        "handlers": handlers,
        "root": {"level": app.config["LOG_LEVEL"], "handlers": root_handlers},
    })


def create_runtime_directories(app: Flask) -> None:
    """Create directories used for local application state."""
    for directory in (Path(app.instance_path), Path(app.config["STORAGE_PATH"])):
        directory.mkdir(parents=True, exist_ok=True)


def initialize_extensions(app: Flask) -> None:
    """Initialize Flask extensions without binding them globally."""
    db.init_app(app)
    migrate.init_app(app, db)
    bootstrap.init_app(app)
    csrf.init_app(app)


def import_models() -> None:
    """Import models so Flask-Migrate can discover their metadata."""
    from app import models  # noqa: F401


def register_routes(app: Flask) -> None:
    """Register application route blueprints."""
    for blueprint in (main, annotations, active_learning, models, datasets, projects, training):
        app.register_blueprint(blueprint)


def register_error_handlers(app: Flask) -> None:
    """Register application error handlers."""
    app.register_blueprint(errors)