"""Primary web routes."""

from flask import Blueprint, render_template

main = Blueprint("main", __name__)


@main.get("/")
def home() -> str:
    """Render the application home page."""
    return render_template("home.html")
