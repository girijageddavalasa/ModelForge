"""Machine-learning plugin registration."""

from app.ml.registry import registry
from app.ml.tabular import GradientBoostingPlugin, LinearModelPlugin, RandomForestPlugin

for plugin_class in (LinearModelPlugin, RandomForestPlugin, GradientBoostingPlugin):
    registry.register(plugin_class())

__all__ = ["registry"]
