# Use a specific, stable Python version
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during the build
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy the rest of your application code into the container
COPY . .

# IMPORTANT: Fix line endings in the entrypoint script
RUN sed -i 's/\r$//' ./entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x ./entrypoint.sh

# Set the entrypoint script as the container's startup command
ENTRYPOINT ["./entrypoint.sh"]

