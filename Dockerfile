# Use a specific, stable Python version
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install system dependencies that might be needed for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during the build
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy the rest of your application code into the container
COPY . .

# Your start command from the Procfile. Railway will inject the $PORT.
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "180", "backend.app:app", "--bind", "0.0.0.0:$PORT"]