#!/bin/sh
# This script correctly starts the Gunicorn server,
# using the PORT variable provided by Railway.
set -e

exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker --timeout 180 backend.app:app --bind 0.0.0.0:$PORT