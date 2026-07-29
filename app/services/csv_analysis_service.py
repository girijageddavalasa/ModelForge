"""CSV data-quality analysis and recommendation services."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import current_app

from app.extensions import db
from app.models import Dataset, DatasetVersion
from app.services.dataset_service import DatasetValidationError

LOGGER = logging.getLogger(__name__)
CORRELATION_THRESHOLD = 0.85
HIGH_CARDINALITY_MIN = 20
IDENTIFIER_UNIQUE_RATIO = 0.95


def analyze_dataset(dataset: Dataset, target_column: str | None = None) -> dict[str, Any]:
    """Analyze a stored tabular dataset and persist structured findings."""
    if dataset.dataset_type != "tabular":
        raise DatasetValidationError("CSV analysis is available only for tabular datasets.")
    version = _current_version(dataset)
    csv_path = _csv_path(dataset, version)
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise DatasetValidationError("The stored CSV file could not be analyzed.") from error
    if frame.empty:
        raise DatasetValidationError("The CSV file has no data rows.")
    if target_column and target_column not in frame.columns:
        raise DatasetValidationError("The selected target column does not exist.")

    rows, columns = frame.shape
    missing = {str(column): int(count) for column, count in frame.isna().sum().items() if count}
    duplicate_rows = int(frame.duplicated().sum())
    numeric_columns = [str(column) for column in frame.select_dtypes(include=np.number).columns]
    categorical_columns = [str(column) for column in frame.columns if str(column) not in numeric_columns]
    unique_counts = {str(column): int(frame[column].nunique(dropna=True)) for column in frame.columns}
    constant_columns = [column for column, count in unique_counts.items() if count <= 1]
    identifier_columns = [
        str(column)
        for column in frame.columns
        if rows > 1
        and unique_counts[str(column)] / rows >= IDENTIFIER_UNIQUE_RATIO
        and (str(column).lower() == "id" or str(column).lower().endswith("_id"))
    ]
    high_cardinality = [
        column
        for column in categorical_columns
        if unique_counts[column] >= HIGH_CARDINALITY_MIN
        and unique_counts[column] / rows >= 0.5
    ]
    correlations = _correlations(frame[numeric_columns]) if len(numeric_columns) > 1 else []
    outliers = _outliers(frame, numeric_columns)
    class_balance = _class_balance(frame, target_column)
    recommendations = _recommendations(
        rows, missing, duplicate_rows, constant_columns, identifier_columns,
        high_cardinality, correlations, outliers, class_balance,
    )
    analysis: dict[str, Any] = {
        "row_count": rows,
        "column_count": columns,
        "target_column": target_column,
        "data_types": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": missing,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "constant_columns": constant_columns,
        "identifier_columns": identifier_columns,
        "high_cardinality_columns": high_cardinality,
        "correlations": correlations,
        "outliers": outliers,
        "class_balance": class_balance,
        "recommendations": recommendations,
    }
    metadata = dict(version.metadata_json or {})
    metadata["analysis"] = analysis
    version.metadata_json = metadata
    version.record_count = rows
    db.session.commit()
    LOGGER.info("Analyzed dataset %s: %s rows, %s columns", dataset.id, rows, columns)
    return analysis


def get_analysis(dataset: Dataset) -> dict[str, Any] | None:
    """Return the most recently persisted analysis, if any."""
    return _current_version(dataset).metadata_json.get("analysis")


def _current_version(dataset: Dataset) -> DatasetVersion:
    """Return the newest dataset version."""
    if not dataset.versions:
        raise DatasetValidationError("The dataset has no version to analyze.")
    return max(dataset.versions, key=lambda version: version.version_number)


def _csv_path(dataset: Dataset, version: DatasetVersion) -> Path:
    """Resolve the stored CSV while enforcing the configured storage root."""
    files = version.metadata_json.get("files", [])
    csv_names = [name for name in files if Path(name).suffix.lower() == ".csv"]
    if len(csv_names) != 1:
        raise DatasetValidationError("The dataset does not reference exactly one CSV file.")
    root = Path(current_app.config["STORAGE_PATH"]).resolve()
    path = (root / dataset.storage_path / csv_names[0]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise DatasetValidationError("Unsafe stored dataset path.") from error
    if not path.is_file():
        raise DatasetValidationError("The stored CSV file is missing.")
    return path


def _correlations(numeric: pd.DataFrame) -> list[dict[str, Any]]:
    """Return unique highly correlated numeric column pairs."""
    matrix = numeric.corr(numeric_only=True)
    findings: list[dict[str, Any]] = []
    for index, left in enumerate(matrix.columns):
        for right in matrix.columns[index + 1:]:
            value = matrix.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= CORRELATION_THRESHOLD:
                findings.append({"left": str(left), "right": str(right), "value": round(float(value), 4)})
    return findings


def _outliers(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    """Count IQR outliers for each numeric column."""
    findings: dict[str, dict[str, Any]] = {}
    for column in columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        count = 0 if iqr == 0 else int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        findings[column] = {"count": count, "percentage": round(count / len(frame) * 100, 2)}
    return findings


def _class_balance(frame: pd.DataFrame, target: str | None) -> dict[str, Any] | None:
    """Summarize target frequencies and imbalance for classification-like targets."""
    if not target:
        return None
    counts = frame[target].value_counts(dropna=False)
    if counts.empty or len(counts) > 50:
        return None
    majority = int(counts.max())
    minority = int(counts.min())
    return {
        "counts": {str(label): int(count) for label, count in counts.items()},
        "minority_majority_ratio": round(minority / majority, 4) if majority else 0,
        "imbalanced": bool(majority and minority / majority < 0.5),
    }


def _recommendations(
    rows: int,
    missing: dict[str, int],
    duplicates: int,
    constants: list[str],
    identifiers: list[str],
    high_cardinality: list[str],
    correlations: list[dict[str, Any]],
    outliers: dict[str, dict[str, Any]],
    class_balance: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Translate measured findings into human-readable advisory actions."""
    items: list[dict[str, str]] = []
    for column, count in missing.items():
        items.append(_item(column, "Missing values", f"{count} ({count / rows * 100:.2f}%)", "Missing values can prevent model training.", "Review and choose an imputation strategy."))
    if duplicates:
        items.append(_item("All columns", "Duplicate rows", str(duplicates), "Duplicates can bias evaluation and training.", "Review and consider removing exact duplicates."))
    for column in constants:
        items.append(_item(column, "Constant column", "1 or fewer unique values", "The column adds no predictive variation.", "Consider excluding this column."))
    for column in identifiers:
        items.append(_item(column, "Identifier-like column", "At least 95% unique", "Identifiers often cause memorization without generalization.", "Review and consider excluding this column."))
    for column in high_cardinality:
        items.append(_item(column, "High cardinality", "Many unique categories", "One-hot encoding may create too many features.", "Review encoding or group rare categories."))
    for pair in correlations:
        items.append(_item(f"{pair['left']}, {pair['right']}", "High correlation", str(pair["value"]), "The columns contain similar numeric information.", "Review whether both columns are needed."))
    for column, result in outliers.items():
        if result["count"]:
            items.append(_item(column, "Potential outliers", f"{result['count']} ({result['percentage']}%)", "Extreme values may influence some models.", "Review values before choosing treatment."))
    if class_balance and class_balance["imbalanced"]:
        items.append(_item("Target", "Class imbalance", str(class_balance["minority_majority_ratio"]), "Minority classes may be predicted poorly.", "Use stratification and review class-aware metrics."))
    return items


def _item(column: str, issue: str, measured: str, explanation: str, action: str) -> dict[str, str]:
    """Build one consistently shaped recommendation."""
    return {"column": column, "issue": issue, "measured_value": measured, "explanation": explanation, "recommended_action": action}
