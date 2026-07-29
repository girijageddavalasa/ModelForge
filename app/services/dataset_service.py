"""Secure local dataset ingestion services."""

from __future__ import annotations

import csv
import io
import logging
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from flask import current_app
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from app.extensions import db
from app.models import Dataset, DatasetVersion, Project

LOGGER = logging.getLogger(__name__)
CSV_EXTENSIONS = {".csv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ZIP_EXTENSIONS = {".zip"}


class DatasetValidationError(ValueError):
    """Raised when uploaded dataset content is unsafe or invalid."""


class DatasetNotFoundError(LookupError):
    """Raised when a dataset does not exist."""


@dataclass(frozen=True)
class PreparedFile:
    """A validated file ready for storage."""

    name: str
    content: bytes


def list_project_datasets(project_id: int) -> list[Dataset]:
    """List datasets for one project, newest first."""
    statement = select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
    return list(db.session.scalars(statement))


def get_dataset(dataset_id: int) -> Dataset:
    """Return a dataset or raise a domain-specific error."""
    dataset = db.session.get(Dataset, dataset_id)
    if dataset is None:
        raise DatasetNotFoundError(f"Dataset {dataset_id} was not found.")
    return dataset


def upload_dataset(project: Project, name: str, files: list[FileStorage]) -> Dataset:
    """Validate, store, and register a CSV or image dataset."""
    clean_name = name.strip()
    if not clean_name:
        raise DatasetValidationError("Dataset name is required.")
    if len(clean_name) > 255:
        raise DatasetValidationError("Dataset name must be 255 characters or fewer.")
    active_files = [uploaded for uploaded in files if uploaded and uploaded.filename]
    if not active_files:
        raise DatasetValidationError("Choose at least one file.")
    max_files = int(current_app.config["MAX_UPLOAD_FILES"])
    if len(active_files) > max_files:
        raise DatasetValidationError(f"A maximum of {max_files} files can be uploaded at once.")

    prepared, dataset_type = _prepare_files(project.task_type, active_files)
    storage_root = Path(current_app.config["STORAGE_PATH"]).resolve()
    dataset_root = storage_root / "datasets" / str(project.id) / uuid.uuid4().hex
    _assert_within(storage_root, dataset_root)
    try:
        dataset_root.mkdir(parents=True, exist_ok=False)
        for item in prepared:
            destination = dataset_root / item.name
            _assert_within(dataset_root, destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(item.content)

        relative_path = dataset_root.relative_to(storage_root).as_posix()
        dataset = Dataset(
            project_id=project.id,
            name=clean_name,
            dataset_type=dataset_type,
            storage_path=relative_path,
            status="ready",
        )
        dataset.versions.append(
            DatasetVersion(
                version_number=1,
                record_count=len(prepared),
                labelled_count=0,
                metadata_json={"files": [item.name for item in prepared]},
            )
        )
        db.session.add(dataset)
        db.session.commit()
        LOGGER.info("Stored dataset %s for project %s with %s files", dataset.id, project.id, len(prepared))
        return dataset
    except Exception:
        db.session.rollback()
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        raise


def _prepare_files(task_type: str, uploads: list[FileStorage]) -> tuple[list[PreparedFile], str]:
    """Validate files based on the owning project's task type."""
    if task_type in {"tabular_classification", "tabular_regression"}:
        if len(uploads) != 1 or Path(uploads[0].filename or "").suffix.lower() not in CSV_EXTENSIONS:
            raise DatasetValidationError("Tabular projects require exactly one CSV file.")
        return [_prepare_csv(uploads[0])], "tabular"

    if task_type == "object_detection":
        if len(uploads) == 1 and Path(uploads[0].filename or "").suffix.lower() in ZIP_EXTENSIONS:
            return _prepare_zip(uploads[0]), "images"
        return _prepare_images(uploads), "images"
    raise DatasetValidationError("The project has an unsupported task type.")


def _read_upload(upload: FileStorage) -> bytes:
    """Read an uploaded file and reject empty content."""
    content = upload.read()
    if not content:
        raise DatasetValidationError(f"{upload.filename or 'File'} is empty.")
    return content


def _safe_name(filename: str) -> str:
    """Return a normalized basename or reject an unsafe name."""
    if Path(filename).name != filename or PurePosixPath(filename).name != filename:
        raise DatasetValidationError(f"Unsafe file path: {filename}.")
    clean_name = secure_filename(filename)
    if not clean_name:
        raise DatasetValidationError("A file has an invalid name.")
    return clean_name


def _prepare_csv(upload: FileStorage) -> PreparedFile:
    """Validate basic CSV readability without performing Stage 4 analysis."""
    name = _safe_name(upload.filename or "")
    content = _read_upload(upload)
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DatasetValidationError("CSV files must use UTF-8 encoding.") from error
    try:
        rows = csv.reader(io.StringIO(text))
        header = next(rows)
    except (csv.Error, StopIteration) as error:
        raise DatasetValidationError("The CSV file is invalid or has no header row.") from error
    if not header or any(not column.strip() for column in header):
        raise DatasetValidationError("Every CSV column must have a non-empty header.")
    if len(set(header)) != len(header):
        raise DatasetValidationError("CSV column names must be unique.")
    try:
        if next(rows, None) is None:
            raise DatasetValidationError("The CSV file must contain at least one data row.")
    except csv.Error as error:
        raise DatasetValidationError("The CSV file could not be parsed.") from error
    return PreparedFile(name=name, content=content)


def _prepare_images(uploads: list[FileStorage]) -> list[PreparedFile]:
    """Validate image extensions, names, duplicates, and image content."""
    prepared: list[PreparedFile] = []
    seen: set[str] = set()
    for upload in uploads:
        name = _safe_name(upload.filename or "")
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            raise DatasetValidationError(f"Unsupported image type: {name}.")
        lowered = name.casefold()
        if lowered in seen:
            raise DatasetValidationError(f"Duplicate file name: {name}.")
        seen.add(lowered)
        content = _read_upload(upload)
        _validate_image(content, name)
        prepared.append(PreparedFile(name=name, content=content))
    return prepared


def _prepare_zip(upload: FileStorage) -> list[PreparedFile]:
    """Safely validate and read images from a ZIP archive."""
    archive_content = _read_upload(upload)
    max_files = int(current_app.config["MAX_UPLOAD_FILES"])
    max_size = int(current_app.config["MAX_EXTRACTED_SIZE"])
    try:
        with zipfile.ZipFile(io.BytesIO(archive_content)) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if not members:
                raise DatasetValidationError("The ZIP archive contains no files.")
            for member in members:
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise DatasetValidationError(f"Unsafe path in ZIP archive: {member.filename}.")
            image_members = [
                member for member in members
                if not _is_ignored_archive_metadata(PurePosixPath(member.filename))
            ]
            if not image_members:
                raise DatasetValidationError("The ZIP archive contains no images.")
            if len(image_members) > max_files:
                raise DatasetValidationError(f"The ZIP archive exceeds the {max_files}-file limit.")
            if sum(member.file_size for member in image_members) > max_size:
                raise DatasetValidationError("The ZIP archive is too large when extracted.")
            prepared: list[PreparedFile] = []
            seen: set[str] = set()
            for member in image_members:
                path = PurePosixPath(member.filename)
                name = _safe_name(path.name)
                if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                    raise DatasetValidationError(f"Unsupported file in ZIP archive: {member.filename}.")
                if name.casefold() in seen:
                    raise DatasetValidationError(f"Duplicate image name in ZIP archive: {name}.")
                seen.add(name.casefold())
                content = archive.read(member)
                _validate_image(content, name)
                prepared.append(PreparedFile(name=name, content=content))
            return prepared
    except zipfile.BadZipFile as error:
        raise DatasetValidationError("The uploaded ZIP archive is corrupted.") from error


def _is_ignored_archive_metadata(path: PurePosixPath) -> bool:
    """Return whether a safe ZIP member is standard macOS metadata."""
    return "__MACOSX" in path.parts or path.name == ".DS_Store" or path.name.startswith("._")

def _validate_image(content: bytes, name: str) -> None:
    """Verify that bytes represent a readable image."""
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise DatasetValidationError(f"Corrupted or invalid image: {name}.") from error


def _assert_within(root: Path, candidate: Path) -> None:
    """Reject any resolved path outside the designated storage root."""
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise DatasetValidationError("Unsafe storage path detected.") from error
