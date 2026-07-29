"""Logistic and linear regression plugin."""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.ml.base_plugin import ModelPlugin


class LinearModelPlugin(ModelPlugin):
    """Linear baseline for classification and regression."""

    name = "linear"
    display_name = "Logistic / Linear Regression"
    supported_tasks = frozenset({"tabular_classification", "tabular_regression"})

    def build_estimator(self, task_type: str, config: dict[str, Any]) -> BaseEstimator:
        """Build the appropriate linear estimator."""
        if task_type == "tabular_classification":
            return LogisticRegression(max_iter=int(config.get("max_iter", 1000)), random_state=int(config.get("random_seed", 42)))
        if task_type == "tabular_regression":
            return LinearRegression()
        raise ValueError(f"Unsupported task type: {task_type}.")
