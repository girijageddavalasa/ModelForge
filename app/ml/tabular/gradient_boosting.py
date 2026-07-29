"""Histogram gradient boosting plugin."""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from app.ml.base_plugin import ModelPlugin


class GradientBoostingPlugin(ModelPlugin):
    """Histogram gradient boosting for tabular tasks."""

    name = "gradient_boosting"
    display_name = "Gradient Boosting"
    supported_tasks = frozenset({"tabular_classification", "tabular_regression"})

    def build_estimator(self, task_type: str, config: dict[str, Any]) -> BaseEstimator:
        """Build a deterministic gradient boosting estimator."""
        parameters = {
            "max_iter": int(config.get("max_iter", 100)),
            "learning_rate": float(config.get("learning_rate", 0.1)),
            "random_state": int(config.get("random_seed", 42)),
        }
        if task_type == "tabular_classification":
            return HistGradientBoostingClassifier(**parameters)
        if task_type == "tabular_regression":
            return HistGradientBoostingRegressor(**parameters)
        raise ValueError(f"Unsupported task type: {task_type}.")
