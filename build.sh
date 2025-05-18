#!/bin/bash
# Build script for Render deployment
set -e  # Exit on error

echo "Starting build process..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements-deploy.txt

# Download NLTK data
echo "Downloading NLTK resources..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Create necessary directories
echo "Creating directories..."
mkdir -p ${DATA_DIR:-./data}
mkdir -p backend/models
mkdir -p backend/embeddings

# Copy sample data to data directory if exists
echo "Setting up data files..."
if [ -d "data" ] && [ "$(ls -A data/*.csv 2>/dev/null)" ]; then
    cp data/*.csv ${DATA_DIR:-./data}/
    echo "Sample data copied to data directory"
else
    echo "No sample data found or data directory empty"
fi

echo "Build completed successfully!"
