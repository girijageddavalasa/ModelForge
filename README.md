# ModelForge Local

ModelForge Local is a local-first AutoML and active learning platform. Stage 1
provides the production-oriented Flask foundation; ML features, APIs,
authentication, and database models are intentionally not included yet.

## Requirements

- Python 3.11 or newer
- `pip`

## Installation

```powershell
cd modelforge-local
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python run.py
```

Open <http://127.0.0.1:5000> in a browser.

On macOS or Linux, activate the environment with
`source .venv/bin/activate` and copy the environment file with
`cp .env.example .env`.

## Configuration

Configuration is loaded from environment variables and an optional `.env`
file. Available environments are `development`, `production`, and `testing`,
selected with `FLASK_ENV`.

For production, set a strong `SECRET_KEY`. SQLite data defaults to
`instance/modelforge.db`, and application artifacts default to `storage/`.
Both directories are created automatically at startup.

## Project structure

```text
app/
  models/       Future SQLAlchemy model modules
  routes/       Future Flask blueprints and thin HTTP handlers
  services/     Business logic
  workers/      Background worker modules
  ml/           Future machine-learning implementation
  templates/    Jinja HTML templates
  static/       CSS and browser assets
  utils/        Shared helpers
instance/       Local SQLite database and instance state
migrations/     Flask-Migrate migration history
storage/        Local datasets, artifacts, and model files
tests/          Automated tests
```

The application factory is `app.create_app`. Extension objects are created
unbound in `app/extensions.py` and initialized by the factory, which keeps the
application testable and avoids import cycles.

## Testing

```powershell
python -m pytest
```

## Docker

Set `SECRET_KEY` in your environment, then run:

```powershell
docker compose up --build
```

## Stage 1 scope

Stage 1 includes only the application foundation, configuration, extension
initialization, logging, Bootstrap shell, SQLite connection settings, and
runtime storage directories. Web handlers are organized as blueprints in
`app/routes/`; routes render responses while future business logic belongs in
`app/services/`. Database models, schema migrations, authentication, APIs, and
machine-learning features are intentionally deferred to later stages.
