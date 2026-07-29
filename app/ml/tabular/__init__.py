"""Built-in tabular model plugins."""

from app.ml.tabular.gradient_boosting import GradientBoostingPlugin
from app.ml.tabular.linear import LinearModelPlugin
from app.ml.tabular.random_forest import RandomForestPlugin

__all__ = ["GradientBoostingPlugin", "LinearModelPlugin", "RandomForestPlugin"]
