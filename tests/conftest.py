"""Shared pytest fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from flask import Flask

from app import create_app
from app.extensions import db


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Flask]:
    """Create an isolated application with an in-memory database and storage."""
    application = create_app("testing")
    application.config["STORAGE_PATH"] = tmp_path / "storage"
    application.config["MAX_UPLOAD_FILES"] = 20
    application.config["MAX_EXTRACTED_SIZE"] = 10 * 1024 * 1024
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app: Flask):
    """Return a Flask test client."""
    return app.test_client()
