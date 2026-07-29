# ModelForge Local

ModelForge Local is planned as a local-first AutoML and active-learning
platform. Development is deliberately divided into stages so every stage can
be tested before the next one begins.

## Current implementation status

**Completed: Stage 2 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Database & Project Management.**

The repository currently includes:

- Flask application factory (`app.create_app`)
- Development, testing, and production configuration classes
- `.env` environment-variable loading
- Flask-SQLAlchemy and Flask-Migrate initialization
- SQLite connection configuration
- Bootstrap 5 integration
- Console logging
- Thin route blueprints for the homepage and error handlers
- Custom 404 and 500 pages
- Automatic creation of `instance/` and `storage/`
- Pytest foundation tests
- Optional Docker configuration

The following features from the overall architecture are **not implemented
yet** because they belong to later stages:

- CSV analysis and preprocessing (Stages 4ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ5)
- AutoML, workers, predictions, and model versioning (Stages 6ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ8)
- Image annotation and YOLO (Stages 9ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ10)
- Active learning (Stage 11)
- Final production hardening (Stage 12)

Do not expect project creation, dataset upload, model training, annotations, or
prediction APIs in the Stage 1 application.

## Stage 2 features

Stage 2 adds:

- Core SQLAlchemy entities: `Project`, `Dataset`, `DatasetVersion`,
  `Annotation`, `TrainingJob`, and `ModelVersion`
- Foreign keys, relationships, uniqueness rules, checks, and indexes
- A generated Flask-Migrate revision for the complete metadata schema
- Dashboard and Project list, create, detail, edit, and delete pages
- A typed `project_service` containing validation and database writes
- Bootstrap navigation, flash messages, empty states, and project summaries

Initialize or upgrade a local database after installing dependencies:

```text
python -m flask --app run.py db upgrade
```

Then start the server with `python run.py`. From the dashboard, select **New
project**, enter a name, select one of the three supported task types, and save.
## Stage 3 features

Stage 3 adds secure local dataset ingestion:

- One UTF-8 CSV file for tabular classification or regression projects
- Multiple JPG, JPEG, PNG, WEBP, or BMP files for object-detection projects
- One ZIP archive containing images for object-detection projects
- Extension, empty-file, CSV-header, data-row, image-content, duplicate-name,
  request-size, archive-size, archive-count, and unsafe-path validation
- Project-specific immutable directories under `storage/datasets/`
- Dataset and initial DatasetVersion records after successful storage
- Dataset list, upload, and detail pages
- Cleanup of partially written files and database rollback on failure

To test manually, create a project, open its **Datasets** page, select **Upload
dataset**, and choose files appropriate to the project's task type. Successful
uploads show an inventory and relative storage path. Stage 3 validates only
safe ingestion; statistical CSV analysis begins in Stage 4.
## Stage 4 features

Stage 4 adds on-demand CSV quality analysis for tabular datasets:

- Row and column counts, inferred data types, and column roles
- Missing-value and duplicate-row counts
- Numeric, categorical, constant, identifier-like, and high-cardinality columns
- IQR-based basic outlier statistics
- Highly correlated numeric pairs (absolute correlation of at least 0.85)
- Optional target selection and classification-style balance summary
- Structured findings persisted in DatasetVersion metadata
- Human-readable recommendations with measured values and explanations
- A Bootstrap analysis report linked from each tabular dataset

Open a tabular dataset and select **Analyze CSV**. Optionally enter a target
column, then run the analysis. Recommendations are advisory: Stage 4 never
removes columns or changes data. Approval and preprocessing begin in Stage 5.
## Stage 5 features

Stage 5 adds a user-approved tabular preprocessing configuration:

- Approve or reject every Stage 4 recommendation
- Required target-column selection
- Approved exclusion of constant and identifier-like feature columns
- Persisted decisions and pipeline configuration in DatasetVersion metadata
- Median numeric imputation and standard scaling
- Most-frequent categorical imputation and unknown-safe one-hot encoding
- A reusable scikit-learn `Pipeline` containing a `ColumnTransformer`
- A Bootstrap approval page with the measured issue and rationale

From a completed CSV analysis, select **Review preprocessing**, choose the
target, decide every recommendation, and approve the configuration. This stage
constructs preprocessing only; model selection and training begin in Stage 6.
## Stage 6 features

Stage 6 adds plugin-based tabular AutoML:

- A stable model-plugin interface and registry
- Logistic Regression / Linear Regression baseline plugin
- Random Forest classifier and regressor plugin
- Histogram Gradient Boosting classifier and regressor plugin
- Selection of one or multiple candidate models
- Deterministic train/test splitting with classification stratification when safe
- Classification metrics: accuracy, weighted precision, recall, F1, confusion
  matrix, and binary ROC-AUC when available
- Regression metrics: MAE, MSE, RMSE, and R-squared
- Complete preprocessing-and-model pipelines exported as `.joblib`
- TrainingJob records containing configuration, progress, results, and paths
- Bootstrap candidate configuration and metric-comparison pages

Open an approved preprocessing configuration and select **Train models**.
Stage 6 executes training synchronously for a simple verified baseline. Stage 7
will move this same service boundary into a local background process and expose
progress polling.
## Stage 7 features

Stage 7 moves model fitting out of the Flask request lifecycle:

- Queued TrainingJob creation before process startup
- Windows-safe `multiprocessing` spawn context and top-level worker target
- A fresh Flask application and SQLAlchemy session inside each worker process
- Persisted queued, running, completed, and failed states
- Candidate-level progress updates and stored worker PID
- Launch-failure and execution-failure persistence
- SQLite WAL mode, foreign-key enforcement, 30-second busy timeout
- `GET /api/training-jobs/<job_id>/status` polling endpoint
- Vanilla JavaScript progress polling and terminal-state page refresh

Starting training now redirects immediately to its progress page. The browser
polls once per second while the local worker fits candidates. The API returns
`id`, `status`, `progress`, `error_message`, and `result_url`. This local process
boundary can later be replaced by a distributed queue without changing model
plugins.
## Stage 8 features

Stage 8 adds an immutable tabular model registry and prediction workflow:

- One ModelVersion per successfully trained candidate
- Monotonic project-level version numbers and idempotent registration
- Exactly one user-selected active version per project through service logic
- Metrics/configuration detail and side-by-side comparison pages
- Safe `.joblib` artifact downloads constrained to local storage
- Browser JSON prediction tester implemented with Vanilla JavaScript
- Single-record and batch tabular prediction
- Class probabilities when the estimator supports them
- Configurable maximum prediction batch size
- Structured validation errors and processing time

Prediction endpoint:

```text
POST /api/models/<model_id>/predict/tabular
Content-Type: application/json

{"age": 35, "city": "A"}
```

A non-empty array of objects is also accepted. Responses include the model ID,
version, model name, processing time, predictions, and probabilities when
available. Image prediction remains deferred until a YOLO model exists.
## Architecture

ModelForge Local uses a layered Flask architecture:

```text
Browser
   |
   v
Routes / Jinja templates   <- HTTP handling and Bootstrap pages
   |
   v
Services                   <- business logic added in later stages
   |
   +-----------> SQLAlchemy / SQLite metadata
   |
   +-----------> Local storage for large files and model artifacts
```

Routes should remain thin. Future business rules belong in `app/services/`,
while future database entities belong in `app/models/`.

## What SQLite is used for

SQLite is the local relational database for ModelForge Local. It stores
structured application metadata such as projects, dataset records, annotation
records, training jobs, and model-version records as those features are added
in later stages.

SQLite does **not** store large CSV files, images, or trained model files. Those
belong in the local `storage/` directory; SQLite stores paths and metadata that
refer to them.

SQLite is a good default for this project because:

- Python includes SQLite support.
- No separate database server must be installed or started.
- The database is a single local file: `instance/modelforge.db`.
- It suits a local, single-user application.
- SQLAlchemy provides an abstraction that can support PostgreSQL later.

Stage 2 creates the core application tables through Flask-Migrate. Apply the latest migration with `python -m flask --app run.py db upgrade`. The database is stored at `instance/modelforge.db`.

## Prerequisites

Install:

- Python 3.11 or newer
- Git
- `pip` (normally installed with Python)
- A modern browser

Docker is optional. PostgreSQL, Redis, Node.js, React, and a GPU are not
required for Stage 1.

Check Python and Git:

```text
python --version
git --version
```

## Install and run on Windows PowerShell

```powershell
git clone https://github.com/girijageddavalasa/modelforge-local.git
cd modelforge-local
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python run.py
```

Open <http://127.0.0.1:5000> or <http://localhost:5000>. You should see a
Bootstrap page containing the title **ModelForge Local**. Stop the server with
`Ctrl+C`.

If PowerShell blocks activation, either adjust its execution policy for your
user or run commands directly with `.venv\Scripts\python.exe`.

## Install and run on Linux or macOS

```bash
git clone https://github.com/girijageddavalasa/modelforge-local.git
cd modelforge-local
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python run.py
```

Open <http://127.0.0.1:5000> and stop the server with `Ctrl+C`.

## How to test the application yourself

Activate the virtual environment and run:

```text
python -m pytest -v
```

Expected result: all tests pass. The current Stage 8 suite contains 38 tests.

The tests verify the full foundation through worker execution plus immutable model registration, exclusive activation, comparison, downloads, input validation, and real single/batch predictions.

You can also verify imports directly:

```text
python -c "from app import create_app; app = create_app('testing'); print(app.name)"
```

Expected output includes `app` and an initialization log message.

For a manual browser test:

1. Run `python run.py`.
2. Visit <http://127.0.0.1:5000>.
3. Confirm that **ModelForge Local** is displayed.
4. Visit <http://127.0.0.1:5000/does-not-exist>.
5. Confirm that the custom 404 page is displayed.
6. Press `Ctrl+C` in the terminal.

## Configuration

Copy `.env.example` to `.env`. Supported values include:

```dotenv
FLASK_ENV=development
SECRET_KEY=replace-with-a-long-random-value
DATABASE_URL=sqlite:///instance/modelforge.db
STORAGE_PATH=storage
LOG_LEVEL=INFO
```

Never commit the real `.env` file. A strong `SECRET_KEY` is mandatory when
`FLASK_ENV=production`.

## Project structure

```text
modelforge-local/
|-- app/
|   |-- __init__.py       Application factory and initialization
|   |-- config.py         Environment-specific configuration
|   |-- extensions.py     Unbound Flask extensions
|   |-- models/           SQLAlchemy metadata entities
|   |-- routes/           Thin HTTP blueprints
|   |-- services/         Future business logic
|   |-- workers/          Future background jobs
|   |-- ml/               Future ML plugins
|   |-- templates/        Jinja and Bootstrap pages
|   |-- static/           CSS and future Vanilla JavaScript
|   `-- utils/            Future shared utilities
|-- instance/             Future SQLite database and local instance data
|-- migrations/           Future Flask-Migrate revision history
|-- storage/              Future datasets and model artifacts
|-- tests/                Automated tests
|-- requirements.txt      Pinned Python dependencies
|-- run.py                Local application entry point
|-- setup.py              Python package metadata
|-- .env.example          Safe configuration example
|-- Dockerfile            Optional container image
`-- docker-compose.yml    Optional local container service
```

The current `setup.py` is package metadata; application startup remains
`python run.py`.

## Optional Docker run

Set `SECRET_KEY` in your shell, then run:

```text
docker compose up --build
```

Docker is not required for normal local development.

## Publish to your GitHub account

First create an empty repository named `modelforge-local` under the GitHub
account `girijageddavalasa`. Do not initialize it with another README, license,
or `.gitignore`. Then, from this project directory, run:

```text
git branch -M main
git remote add origin https://github.com/girijageddavalasa/modelforge-local.git
git push -u origin main
```

If `origin` already exists, inspect it with `git remote -v` and change it with:

```text
git remote set-url origin https://github.com/girijageddavalasa/modelforge-local.git
```

GitHub may ask you to authenticate through a browser or personal access token.

## Troubleshooting

- `ModuleNotFoundError`: activate `.venv` and rerun
  `python -m pip install -r requirements.txt`.
- `python` is not recognized: reinstall Python 3.11+ and enable the installer
  option that adds Python to `PATH`; on Linux/macOS, try `python3`.
- Port 5000 is busy: stop the other process using that port before starting the
  application.
- Production refuses to start: set a non-default `SECRET_KEY`.
- No SQLite file exists yet: this is expected in Stage 1 because models and
  migrations begin in Stage 2.
