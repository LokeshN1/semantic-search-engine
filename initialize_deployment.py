#!/usr/bin/env python
"""
Initialization script for deployment.
This script downloads required NLTK resources and Spacy models.
"""

import os
import sys
import nltk
import spacy
import subprocess
import importlib.util

def check_package(package_name):
    """Check if a package is installed."""
    return importlib.util.find_spec(package_name) is not None

def main():
    print("Initializing deployment...")
    
    # Download NLTK resources
    print("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Download spaCy model
    print("Downloading spaCy model...")
    subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), "backend", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Initialization complete!")
    
if __name__ == "__main__":
    main() 