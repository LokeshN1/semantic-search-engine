#!/bin/bash
# Build script for Render deployment
set -e  # Exit on error

echo "Starting build process..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements-deploy.txt

# Download NLTK data - minimal set for free tier
echo "Downloading minimal NLTK resources..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Only download the small spaCy model for free tier
echo "Downloading small spaCy model..."
python -m spacy download en_core_web_sm

# Create necessary directories
echo "Creating directories..."
mkdir -p ${DATA_DIR:-./data}
mkdir -p backend/models
mkdir -p backend/embeddings

# Create a minimal sample dataset for faster startup
echo "Creating minimal sample dataset..."
cat > ${DATA_DIR}/minimal_sample.csv << EOF
id,text,sentiment
1,"This is a very positive sample text for testing.",positive
2,"This sample text is rather negative in tone.",negative
3,"This content is quite neutral in its presentation.",neutral
4,"Amazing service and excellent product quality!",positive
5,"Terrible experience with poor customer support.",negative
EOF
echo "Created minimal sample dataset at ${DATA_DIR}/minimal_sample.csv"

# Disable BERT by default on free tier
echo "Configuring app for free tier..."

# Copy sample data to data directory if exists and if DATA_DIR is different from ./data
echo "Setting up data files..."
if [ -d "data" ] && [ -n "$(ls -A data/*.csv 2>/dev/null)" ]; then
    # Only copy if DATA_DIR is not the default ./data location
    if [ "${DATA_DIR}" != "./data" ] && [ "${DATA_DIR}" != "data" ]; then
        # Copy only a small subset of data files to avoid memory issues
        head -n 1000 data/*.csv > ${DATA_DIR}/sample_data.csv
        echo "Small sample data created at ${DATA_DIR}/sample_data.csv"
    else
        echo "Using sample data in default data directory"
    fi
else
    echo "No sample data found or data directory empty"
fi

# Clean up any cached data to save space
echo "Cleaning up to save space..."
rm -rf ~/.cache/huggingface

echo "Build completed successfully!"
