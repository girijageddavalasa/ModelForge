"""Image gallery and annotation HTTP routes."""

from __future__ import annotations

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, send_file, url_for

from app.extensions import csrf
from app.services import active_learning_service, annotation_service, dataset_service
from app.services.annotation_service import AnnotationValidationError
from app.services.dataset_service import DatasetNotFoundError, DatasetValidationError

annotations = Blueprint("annotations", __name__)


@annotations.get("/datasets/<int:dataset_id>/images")
def gallery(dataset_id: int) -> str:
    """Display filterable image cards and annotation status."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
        status = request.args.get("status")
        images = annotation_service.image_inventory(dataset, status)
        classes = annotation_service.get_classes(dataset)
    except DatasetNotFoundError:
        abort(404)
    except DatasetValidationError as error:
        flash(str(error), "danger")
        return redirect(url_for("datasets.detail", dataset_id=dataset_id))
    return render_template("annotations/gallery.html", dataset=dataset, images=images, classes=classes, status=status)


@annotations.post("/datasets/<int:dataset_id>/classes")
def save_classes(dataset_id: int) -> str:
    """Persist newline- or comma-separated object class names."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
        raw = request.form.get("classes", "").replace(",", "\n")
        annotation_service.save_classes(dataset, raw.splitlines())
        flash("Object classes saved.", "success")
    except DatasetNotFoundError:
        abort(404)
    except (DatasetValidationError, AnnotationValidationError) as error:
        flash(str(error), "danger")
    return redirect(url_for("annotations.gallery", dataset_id=dataset_id))


@annotations.get("/datasets/<int:dataset_id>/images/<path:filename>")
def image_file(dataset_id: int, filename: str):
    """Serve only images listed in the selected dataset version."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
        path = annotation_service.image_path(dataset, filename)
    except (DatasetNotFoundError, FileNotFoundError):
        abort(404)
    except (DatasetValidationError, AnnotationValidationError):
        abort(400)
    return send_file(path, conditional=True)


@annotations.get("/datasets/<int:dataset_id>/annotate/<path:filename>")
def editor(dataset_id: int, filename: str) -> str:
    """Render the Konva.js annotation editor for one whitelisted image."""
    try:
        dataset = dataset_service.get_dataset(dataset_id)
        annotation_service.image_path(dataset, filename)
        classes = annotation_service.get_classes(dataset)
        previous_name = annotation_service.adjacent_image(dataset, filename, -1)
        next_name = annotation_service.adjacent_image(dataset, filename, 1)
    except (DatasetNotFoundError, FileNotFoundError):
        abort(404)
    except DatasetValidationError as error:
        flash(str(error), "danger")
        return redirect(url_for("datasets.detail", dataset_id=dataset_id))
    return render_template(
        "annotations/editor.html", dataset=dataset, filename=filename, classes=classes,
        previous_name=previous_name, next_name=next_name,
    )


@annotations.get("/api/datasets/<int:dataset_id>/annotations")
def annotation_list(dataset_id: int):
    """Return saved boxes and configured classes for one image."""
    filename = request.args.get("image", "")
    try:
        dataset = dataset_service.get_dataset(dataset_id)
        boxes = annotation_service.list_annotations(dataset, filename)
        classes = annotation_service.get_classes(dataset)
    except DatasetNotFoundError:
        return jsonify({"error": "Dataset not found."}), 404
    except FileNotFoundError:
        return jsonify({"error": "Image not found."}), 404
    except (DatasetValidationError, AnnotationValidationError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"image": filename, "classes": classes, "annotations": boxes})


@annotations.put("/api/datasets/<int:dataset_id>/annotations")
@csrf.exempt
def annotation_save(dataset_id: int):
    """Validate and replace annotations for one image."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON annotation object."}), 400
    try:
        dataset = dataset_service.get_dataset(dataset_id)
        boxes = payload.get("annotations")
        if not isinstance(boxes, list):
            raise AnnotationValidationError("Annotations must be a list.")
        saved = annotation_service.save_annotations(
            dataset,
            str(payload.get("image", "")),
            int(payload.get("image_width", 0)),
            int(payload.get("image_height", 0)),
            boxes,
        )
        active_learning_service.mark_reviewed(dataset, str(payload.get("image", "")))
    except DatasetNotFoundError:
        return jsonify({"error": "Dataset not found."}), 404
    except FileNotFoundError:
        return jsonify({"error": "Image not found."}), 404
    except (TypeError, ValueError, DatasetValidationError, AnnotationValidationError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"saved": len(saved), "annotations": saved})
