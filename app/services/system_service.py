"""Operational readiness checks."""

from __future__ import annotations

from sqlalchemy import text

from app.extensions import db


def database_is_ready() -> bool:
    """Return whether the configured database accepts a trivial query."""
    try:
        db.session.execute(text("SELECT 1"))
    except Exception:
        db.session.rollback()
        return False
    return True