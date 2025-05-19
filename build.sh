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

# Create a minimal sample dataset for faster startup
echo "Creating minimal sample dataset..."
cat > ${DATA_DIR}/minimal_sample.csv << EOF
id,text,sentiment
1,"This is a very positive sample text for testing.",positive
2,"This sample text is rather negative in tone.",negative
3,"I feel neutral about this particular sample.",neutral
4,"Amazing service and great product quality!",positive
5,"Terrible experience with customer support.",negative
EOF
echo "Created minimal sample dataset at ${DATA_DIR}/minimal_sample.csv"

# Copy sample data to data directory if exists and if DATA_DIR is different from ./data
echo "Setting up data files..."
if [ -d "data" ] && [ -n "$(ls -A data/*.csv 2>/dev/null)" ]; then
    # Only copy if DATA_DIR is not the default ./data location
    if [ "${DATA_DIR}" != "./data" ] && [ "${DATA_DIR}" != "data" ]; then
        cp data/*.csv ${DATA_DIR}/
        echo "Sample data copied to ${DATA_DIR} directory"
    else
        echo "Using sample data in default data directory"
    fi
else
    echo "No sample data found or data directory empty"
fi

echo "Build completed successfully!"
