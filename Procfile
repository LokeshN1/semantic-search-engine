web: /bin/sh -c "python initialize_deployment.py && gunicorn -w 1 -k uvicorn.workers.UvicornWorker --timeout 180 backend.app:app --bind 0.0.0.0:$PORT"
