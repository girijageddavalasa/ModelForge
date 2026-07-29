"""Unbound Flask extension instances."""

from flask_bootstrap import Bootstrap5
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for future SQLAlchemy models."""


db = SQLAlchemy(model_class=Base)
migrate = Migrate()
bootstrap = Bootstrap5()
