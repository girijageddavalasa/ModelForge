"""Random forest plugin."""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.ml.base_plugin import ModelPlugin


class RandomForestPlugin(ModelPlugin):
    """Random forest models for classification and regression."""

    name = "random_forest"
    display_name = "Random Forest"
    supported_tasks = frozenset({"tabular_classification", "tabular_regression"})

    def build_estimator(self, task_type: str, config: dict[str, Any]) -> BaseEstimator:
        """Build a deterministic random forest estimator."""
        parameters = {
            "n_estimators": int(config.get("n_estimators", 100)),
            "max_depth": config.get("max_depth"),
            "random_state": int(config.get("random_seed", 42)),
            "n_jobs": -1,
        }
        if task_type == "tabular_classification":
            return RandomForestClassifier(**parameters)
        if task_type == "tabular_regression":
            return RandomForestRegressor(**parameters)
        raise ValueError(f"Unsupported task type: {task_type}.")
