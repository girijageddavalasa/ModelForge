"""Tabular model prediction services."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import current_app

from app.models import ModelVersion
from app.services.model_registry_service import artifact_path


class PredictionValidationError(ValueError):
    """Raised when prediction input is invalid."""


def predict_tabular(model: ModelVersion, payload: Any) -> dict[str, Any]:
    """Validate JSON records and run a stored tabular pipeline."""
    records = _records(payload)
    limit = int(current_app.config["PREDICTION_MAX_BATCH"])
    if len(records) > limit:
        raise PredictionValidationError(f"A maximum of {limit} records can be predicted at once.")
    frame = pd.DataFrame.from_records(records)
    if frame.empty or not len(frame.columns):
        raise PredictionValidationError("Each prediction record must contain feature values.")
    started = perf_counter()
    try:
        pipeline = joblib.load(artifact_path(model))
        predictions = pipeline.predict(frame)
        probabilities = pipeline.predict_proba(frame) if hasattr(pipeline, "predict_proba") else None
    except (KeyError, ValueError, TypeError) as error:
        raise PredictionValidationError(f"Prediction input does not match the model: {error}") from error
    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    results: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        item: dict[str, Any] = {"prediction": _json_value(prediction)}
        if probabilities is not None:
            item["probabilities"] = [round(float(value), 8) for value in probabilities[index]]
        results.append(item)
    return {
        "model_id": model.id,
        "model_version": model.version_number,
        "model_name": model.model_name,
        "processing_time_ms": elapsed_ms,
        "results": results,
    }


def _records(payload: Any) -> list[dict[str, Any]]:
    """Normalize one JSON object or a list of objects."""
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and payload and all(isinstance(item, dict) for item in payload):
        return payload
    raise PredictionValidationError("Send one JSON object or a non-empty list of JSON objects.")


def _json_value(value: Any) -> Any:
    """Convert NumPy scalar outputs into JSON-native values."""
    return value.item() if isinstance(value, np.generic) else value
