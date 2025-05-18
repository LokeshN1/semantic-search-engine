#!/bin/bash
# Build script for Render deployment

# Upgrade pip first
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements-deploy.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet

# Download spaCy model
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p ${DATA_DIR:-./data}
mkdir -p backend/models

# Copy sample data to data directory
cp data/*.csv ${DATA_DIR:-./data}/ || true

echo "Build completed successfully!"
