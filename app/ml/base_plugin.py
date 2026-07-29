"""Abstract interface for ModelForge model plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sklearn.base import BaseEstimator


class ModelPlugin(ABC):
    """Contract implemented by trainable model plugins."""

    name: str
    display_name: str
    supported_tasks: frozenset[str]

    @abstractmethod
    def build_estimator(self, task_type: str, config: dict[str, Any]) -> BaseEstimator:
        """Create an unfitted estimator for a supported task."""
        raise NotImplementedError
