# ModelForge Local

ModelForge Local is a local-first AutoML and active-learning platform for tabular classification, tabular regression, and YOLO object detection. Data, SQLite metadata, model artifacts, and generated datasets remain on the local machine.

**Current status: Stage 11 - Active Learning is complete.** Stage 12 production hardening is not implemented yet.

## Features implemented

- Flask application factory with development, testing, and production configuration
- SQLAlchemy ORM, SQLite, Flask-Migrate, Bootstrap 5, structured logging, and environment loading
- Project CRUD and dashboard
- Secure CSV, image, and ZIP dataset upload with immutable dataset versions
- CSV quality analysis and recommendations
- Approved scikit-learn preprocessing pipelines
- Tabular AutoML for classification and regression
- Multiprocessing training jobs and progress polling
- Prediction APIs, model registry, comparison, activation, and downloads
- Browser-based bounding-box annotation with Konva.js and keyboard shortcuts
- YOLO dataset generation, training, PT artifacts, ONNX export, and ONNX Runtime validation
- Active-model pre-annotation, confidence thresholding, uncertainty-ranked review, correction provenance, reviewed dataset versions, and retraining handoff

## Requirements

- Python 3.11 or newer
- Git
- Enough local disk space for uploaded datasets and trained models

## Install and run

```text
git clone https://github.com/girijageddavalasa/ModelForge.git
cd ModelForge
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
python -m flask --app run.py db upgrade
python run.py
```

On macOS or Linux, activate with `source .venv/bin/activate` and copy the environment file with `cp .env.example .env`.

Open `http://127.0.0.1:5000` in a browser. Do not use Flask's development server for an internet-facing production deployment.

## What SQLite is used for

SQLite is the local relational database stored as a single file. ModelForge uses it for project records, dataset and dataset-version metadata, annotations, training-job status, model versions, configuration, and metrics. Large files are not stored inside SQLite: uploaded datasets and model artifacts live below `storage/`, while the database stores their safe relative paths.

Flask-Migrate manages schema changes. Initialize an existing checkout with:

```text
python -m flask --app run.py db upgrade
```

Back up both the SQLite database under `instance/` and the `storage/` directory to preserve a complete installation.

## Manual testing

1. Start the application with `python run.py` and open the local URL.
2. Create a tabular or object-detection project from the dashboard.
3. Upload a suitable CSV, image collection, or image ZIP.
4. For tabular data, analyze the CSV, approve preprocessing, train candidates, activate a model, and test predictions.
5. For detection, configure classes, draw boxes, train YOLO, and activate the resulting model.
6. Open **Active learning**, choose a confidence threshold, and start pre-annotation.
7. Review images in uncertainty order. Moving, resizing, or relabeling a model box records it as `human_corrected`; untouched predictions retain `model` provenance.
8. After every queued image is reviewed, create the next dataset version and configure YOLO retraining.

Run automated checks from the repository root:

```text
python -m pytest -q
python -m compileall -q app tests
node --check app/static/js/annotation_canvas.js
python -m flask --app run.py db check
```

The current automated suite contains 54 passing tests. YOLO training may download the selected base weights on first use.

## Active-learning workflow

Pre-annotation requires an image dataset with configured classes and an active YOLO model for the same project. A background worker runs predictions only on unreviewed images and stores proposed boxes as pending model annotations. The review queue ranks images by uncertainty (`1 - mean box confidence`); images without detections receive maximum uncertainty. Saving an image accepts its reviewed state. Finalization is blocked until every queued image is reviewed, then creates a new immutable dataset version and redirects to manual YOLO retraining configuration.

## Configuration

Copy `.env.example` to `.env`. Important settings include the Flask environment, secret key, SQLite database URL, upload limits, and storage location. Production mode requires a strong `SECRET_KEY`. Paths are handled with `pathlib` and stored relative to the configured storage root.

## Project structure

```text
app/
  __init__.py       Application factory and blueprint registration
  config.py         Environment-specific configuration
  extensions.py     SQLAlchemy, migrations, and Bootstrap extensions
  models/           SQLAlchemy entities
  routes/           Thin HTTP and JSON endpoint handlers
  services/         Validation and business workflows
  workers/          Multiprocessing job launch and dispatch
  ml/               AutoML and YOLO plugin implementations
  templates/        Bootstrap/Jinja pages
  static/           CSS and vanilla JavaScript
  utils/            Shared helpers
migrations/         Flask-Migrate revisions
storage/            Local datasets and generated model artifacts
instance/           Local SQLite database and instance configuration
tests/              Automated test suite
run.py              Local application entry point
```

Routes translate HTTP requests and responses; services own business rules and persistence workflows. Model/plugin code remains isolated in `app/ml`, and background execution remains in `app/workers`.

## Docker

```text
docker compose up --build
```

The compose configuration mounts persistent local storage. Review `.env` and production secrets before exposing the service outside localhost.

## Push changes to GitHub

The repository remote is `girijageddavalasa/ModelForge`. After a completed stage:

```text
git add .
git commit -m "Stage X: Stage name"
git push origin main
```

Check the result at https://github.com/girijageddavalasa/ModelForge.

## Next stage

Stage 12 will add production hardening: expanded security and testing, deployment logging, documentation, and performance tuning. It is intentionally outside Stage 11.