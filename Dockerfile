# Use a specific, stable Python version
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt stopwords wordnet

COPY . .

# Expose a default port (Docker needs a number here, not $PORT)
EXPOSE 8000

# Use $PORT only in the CMD (Railway will set it at runtime)
CMD exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker --timeout 180 backend.app:app --bind 0.0.0.0:${PORT:-8000}
