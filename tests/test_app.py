"""Application foundation tests."""

from flask import Flask

from app import create_app


def test_create_app() -> None:
    """The factory creates a Flask application."""
    app = create_app("testing")
    assert isinstance(app, Flask)
    assert app.config["TESTING"] is True


def test_home_page() -> None:
    """The home page renders the project title."""
    app = create_app("testing")
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"ModelForge Local" in response.data


def test_not_found_page() -> None:
    """Unknown routes render the custom 404 page."""
    app = create_app("testing")
    response = app.test_client().get("/missing")
    assert response.status_code == 404
    assert b"Page not found" in response.data
