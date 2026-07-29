"""Machine-learning plugin registration."""

from app.ml.registry import registry
from app.ml.tabular import GradientBoostingPlugin, LinearModelPlugin, RandomForestPlugin
from app.ml.vision import YoloPlugin

for plugin_class in (LinearModelPlugin, RandomForestPlugin, GradientBoostingPlugin, YoloPlugin):
    registry.register(plugin_class())

__all__ = ["registry"]
