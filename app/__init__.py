"""Application factory for ModelForge Local."""

from __future__ import annotations

from logging.config import dictConfig
from pathlib import Path

from flask import Flask

from app.config import CONFIGURATIONS, Config, ProductionConfig
from app.extensions import bootstrap, db, migrate
from app.routes.datasets import datasets
from app.routes.errors import errors
from app.routes.main import main
from app.routes.projects import projects
from app.routes.training import training


def create_app(
    config_name: str | None = None,
    config_overrides: dict[str, object] | None = None,
) -> Flask:
    """Create and configure a ModelForge Local Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    selected_config = config_name or Config.environment_name()
    config_class = CONFIGURATIONS.get(selected_config)
    if config_class is None:
        valid_names = ", ".join(sorted(CONFIGURATIONS))
        raise ValueError(
            f"Unknown configuration {selected_config!r}. Choose one of: {valid_names}."
        )

    if config_class is ProductionConfig:
        ProductionConfig.validate()

    app.config.from_object(config_class)
    configure_logging(app)
    create_runtime_directories(app)
    initialize_extensions(app)
    import_models()
    register_routes(app)
    register_error_handlers(app)

    app.logger.info("ModelForge Local initialized with %s configuration", selected_config)
    return app


def configure_logging(app: Flask) -> None:
    """Configure structured console logging for the application."""
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {
                "level": app.config["LOG_LEVEL"],
                "handlers": ["console"],
            },
        }
    )


def create_runtime_directories(app: Flask) -> None:
    """Create directories used for local application state."""
    directories = (Path(app.instance_path), Path(app.config["STORAGE_PATH"]))
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def initialize_extensions(app: Flask) -> None:
    """Initialize Flask extensions without binding them globally."""
    db.init_app(app)
    migrate.init_app(app, db)
    bootstrap.init_app(app)


def import_models() -> None:
    """Import models so Flask-Migrate can discover their metadata."""
    from app import models  # noqa: F401


def register_routes(app: Flask) -> None:
    """Register application route blueprints."""
    app.register_blueprint(main)
    app.register_blueprint(datasets)
    app.register_blueprint(projects)
    app.register_blueprint(training)


def register_error_handlers(app: Flask) -> None:
    """Register application error handlers."""
    app.register_blueprint(errors)
