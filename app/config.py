"""Configuration objects for ModelForge Local."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCE_PATH = PROJECT_ROOT / "instance"

load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """Shared application configuration."""

    SECRET_KEY = os.getenv("SECRET_KEY", "development-only-secret-key")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(INSTANCE_PATH / 'modelforge.db').as_posix()}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    STORAGE_PATH = Path(os.getenv("STORAGE_PATH", PROJECT_ROOT / "storage")).resolve()
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 500 * 1024 * 1024))
    MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", 5000))
    MAX_EXTRACTED_SIZE = int(os.getenv("MAX_EXTRACTED_SIZE", 500 * 1024 * 1024))
    PREDICTION_MAX_BATCH = int(os.getenv("PREDICTION_MAX_BATCH", 1000))
    MAX_ANNOTATIONS_PER_IMAGE = int(os.getenv("MAX_ANNOTATIONS_PER_IMAGE", 1000))

    @staticmethod
    def environment_name() -> str:
        """Return the configured application environment."""
        return os.getenv("FLASK_ENV", "development").lower()


class DevelopmentConfig(Config):
    """Configuration for local development."""

    DEBUG = True


class ProductionConfig(Config):
    """Configuration for production deployments."""

    DEBUG = False
    TESTING = False

    @classmethod
    def validate(cls) -> None:
        """Validate security-sensitive production settings."""
        if cls.SECRET_KEY == "development-only-secret-key":
            raise RuntimeError("SECRET_KEY must be set in production.")


class TestingConfig(Config):
    """Configuration for automated tests."""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False


CONFIGURATIONS: dict[str, type[Config]] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
