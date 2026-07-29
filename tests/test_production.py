"""Production security, health, and configuration tests."""

from __future__ import annotations

import re
from pathlib import Path

from app import create_app
from app.extensions import db


def _hardened_app(tmp_path: Path):
    application = create_app("development", {
        "TESTING": True,
        "WTF_CSRF_ENABLED": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "STORAGE_PATH": tmp_path / "storage",
        "LOG_PATH": None,
        "SECRET_KEY": "test-only-strong-secret",
    })
    with application.app_context():
        db.create_all()
    return application


def test_health_and_security_headers(tmp_path: Path) -> None:
    """Health responses include request correlation and browser defenses."""
    application = _hardened_app(tmp_path)
    client = application.test_client()
    response = client.get("/health/ready", headers={"X-Request-ID": "stage-12-check"})
    assert response.status_code == 200 and response.get_json() == {"status": "ok"}
    assert response.headers["X-Request-ID"] == "stage-12-check"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["Cache-Control"] == "no-store"


def test_browser_forms_require_csrf(tmp_path: Path) -> None:
    """Unsafe browser requests fail without the session-bound CSRF token."""
    application = _hardened_app(tmp_path)
    client = application.test_client()
    assert client.post("/projects/new", data={"name": "Blocked", "task_type": "tabular_classification"}).status_code == 400
    form = client.get("/projects/new")
    token = re.search(rb'name="csrf_token" value="([^"]+)"', form.data)
    assert token is not None
    response = client.post("/projects/new", data={
        "csrf_token": token.group(1).decode(), "name": "Allowed",
        "description": "", "task_type": "tabular_classification",
    })
    assert response.status_code == 302


def test_production_requires_secret_and_honors_overrides(tmp_path: Path) -> None:
    """Production rejects the development secret and accepts explicit safe overrides."""
    try:
        create_app("production", {"SECRET_KEY": "development-only-secret-key", "LOG_PATH": None})
    except RuntimeError as error:
        assert "SECRET_KEY" in str(error)
    else:
        raise AssertionError("Production accepted the development secret")
    try:
        create_app("production", {"SECRET_KEY": "too-short", "LOG_PATH": None})
    except RuntimeError:
        pass
    else:
        raise AssertionError("Production accepted a weak secret")
    application = create_app("production", {
        "SECRET_KEY": "a-production-secret-with-sufficient-entropy",
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "STORAGE_PATH": tmp_path / "storage", "LOG_PATH": None,
    })
    assert application.config["SECRET_KEY"].startswith("a-production")