"""Application foundation tests."""

from flask import Flask
from flask.testing import FlaskClient


def test_create_app(app: Flask) -> None:
    """The fixture uses a factory-created Flask application."""
    assert isinstance(app, Flask)
    assert app.config["TESTING"] is True


def test_home_page(client: FlaskClient) -> None:
    """The home page renders the project title."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"ModelForge Local" in response.data


def test_not_found_page(client: FlaskClient) -> None:
    """Unknown routes render the custom 404 page."""
    response = client.get("/missing")
    assert response.status_code == 404
    assert b"Page not found" in response.data
