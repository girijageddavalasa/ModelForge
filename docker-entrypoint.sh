#!/bin/sh
set -eu
python -m flask --app run.py db upgrade
exec gunicorn --bind 0.0.0.0:5000 --workers "${WEB_WORKERS:-1}" --threads "${WEB_THREADS:-4}" --timeout "${WEB_TIMEOUT:-300}" --access-logfile - --error-logfile - run:app