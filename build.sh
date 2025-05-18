#!/bin/bash
# Build script for Render deployment

set -e  # Exit immediately if a command exits with non-zero status

echo "Starting build process..."

# Upgrade pip first
python -m pip install --upgrade pip

# Install dependencies with specific versions to prevent conflicts
echo "Installing dependencies..."
pip install -r requirements-deploy.txt --no-cache-dir

# Download NLTK data with progress output
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download minimal spaCy model
echo "Downloading spaCy models..."
python -m spacy download en_core_web_sm --no-deps

# Create necessary directories
echo "Creating directories..."
mkdir -p ${DATA_DIR:-./data}
mkdir -p backend/models
mkdir -p backend/models/cache

# Set cache directories for transformers and torch
echo "Setting up cache directories..."
export TRANSFORMERS_CACHE="./backend/models/cache"
export TORCH_HOME="./backend/models/cache"
export HF_HOME="./backend/models/cache"

# Pre-download the smaller model to avoid runtime downloads
echo "Pre-downloading sentence-transformer model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy sample data to data directory
echo "Copying sample data..."
cp -f data/*.csv ${DATA_DIR:-./data}/ 2>/dev/null || true

echo "Build completed successfully!"
