"""Preprocessing approval and pipeline construction services."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from flask import current_app
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.extensions import db
from app.models import Dataset, DatasetVersion
from app.services.dataset_service import DatasetValidationError

LOGGER = logging.getLogger(__name__)
VALID_DECISIONS = {"approve", "reject"}


def get_preprocessing(dataset: Dataset) -> dict[str, Any] | None:
    """Return persisted preprocessing approval metadata."""
    return _current_version(dataset).metadata_json.get("preprocessing")


def save_decisions(
    dataset: Dataset,
    target_column: str,
    submitted: dict[int, str],
) -> dict[str, Any]:
    """Validate and persist a decision for every analysis recommendation."""
    version = _current_version(dataset)
    analysis = version.metadata_json.get("analysis")
    if not analysis:
        raise DatasetValidationError("Run CSV analysis before approving preprocessing.")
    columns = set(analysis["data_types"])
    if target_column not in columns:
        raise DatasetValidationError("Select a valid target column.")
    recommendations = analysis.get("recommendations", [])
    if (
        set(submitted) != set(range(len(recommendations)))
        or any(not value for value in submitted.values())
    ):
        raise DatasetValidationError("Approve or reject every recommendation.")
    if any(value not in VALID_DECISIONS for value in submitted.values()):
        raise DatasetValidationError("A preprocessing decision is invalid.")

    decisions = [
        {
            "recommendation_index": index,
            "decision": submitted[index],
            "issue": recommendation["issue"],
            "column": recommendation["column"],
            "recommended_action": recommendation["recommended_action"],
        }
        for index, recommendation in enumerate(recommendations)
    ]
    excluded = _approved_exclusions(decisions)
    if target_column in excluded:
        raise DatasetValidationError("The target column cannot be excluded.")
    numeric = [column for column in analysis["numeric_columns"] if column != target_column and column not in excluded]
    categorical = [column for column in analysis["categorical_columns"] if column != target_column and column not in excluded]
    if not numeric and not categorical:
        raise DatasetValidationError("No feature columns remain after approved exclusions.")
    preprocessing = {
        "status": "approved",
        "target_column": target_column,
        "decisions": decisions,
        "excluded_columns": sorted(excluded),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "categorical_encoder": "one_hot",
        "scale_numeric": True,
    }
    metadata = dict(version.metadata_json)
    metadata["preprocessing"] = preprocessing
    version.metadata_json = metadata
    db.session.commit()
    LOGGER.info("Saved preprocessing approval for dataset %s", dataset.id)
    return preprocessing


def build_preprocessor(config: dict[str, Any]) -> Pipeline:
    """Build the approved fitted-ready sklearn preprocessing pipeline."""
    if config.get("status") != "approved":
        raise DatasetValidationError("Preprocessing decisions have not been approved.")
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    numeric_features = list(config.get("numeric_features", []))
    categorical_features = list(config.get("categorical_features", []))
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    if not transformers:
        raise DatasetValidationError("No feature columns are configured.")
    transformer = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline(steps=[("preprocessor", transformer)])


def load_features_and_target(dataset: Dataset, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Load the stored CSV and split approved features from the target."""
    version = _current_version(dataset)
    files = version.metadata_json.get("files", [])
    csv_files = [name for name in files if Path(name).suffix.lower() == ".csv"]
    if len(csv_files) != 1:
        raise DatasetValidationError("The dataset does not reference exactly one CSV file.")
    root = Path(current_app.config["STORAGE_PATH"]).resolve()
    path = (root / dataset.storage_path / csv_files[0]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise DatasetValidationError("Unsafe stored dataset path.") from error
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise DatasetValidationError("The stored CSV file could not be loaded.") from error
    target = config["target_column"]
    feature_columns = list(config["numeric_features"]) + list(config["categorical_features"])
    return frame[feature_columns], frame[target]


def _approved_exclusions(decisions: list[dict[str, Any]]) -> set[str]:
    """Determine columns explicitly approved for exclusion."""
    exclusions: set[str] = set()
    exclusion_issues = {"Constant column", "Identifier-like column"}
    for item in decisions:
        if item["decision"] == "approve" and item["issue"] in exclusion_issues:
            exclusions.add(item["column"])
    return exclusions


def _current_version(dataset: Dataset) -> DatasetVersion:
    """Return the newest dataset version."""
    if not dataset.versions:
        raise DatasetValidationError("The dataset has no version.")
    return max(dataset.versions, key=lambda version: version.version_number)
