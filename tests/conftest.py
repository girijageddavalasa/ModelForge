"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from flask import Flask

from app import create_app
from app.extensions import db


@pytest.fixture
def app() -> Iterator[Flask]:
    """Create an isolated application with an in-memory database."""
    application = create_app("testing")
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app: Flask):
    """Return a Flask test client."""
    return app.test_client()
