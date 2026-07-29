"""Unbound Flask extension instances and SQLite connection policy."""

from __future__ import annotations

import sqlite3

from flask_bootstrap import Bootstrap5
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase


@event.listens_for(Engine, "connect")
def configure_sqlite_connection(dbapi_connection: object, _connection_record: object) -> None:
    """Enable integrity and worker-friendly concurrency settings for SQLite."""
    if not isinstance(dbapi_connection, sqlite3.Connection):
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


db = SQLAlchemy(model_class=Base)
migrate = Migrate()
bootstrap = Bootstrap5()
