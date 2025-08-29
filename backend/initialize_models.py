import os
import sys
import time
import pickle
import numpy as np
import shutil
from pathlib import Path
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Download NLTK data
print("Downloading required NLTK data...")
try:
    import nltk
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

# Clean models directory to ensure fresh initialization
MODELS_DIR = os.environ.get("MODELS_DIR", "./backend/models")
print(f"Cleaning models directory: {MODELS_DIR}")
if os.path.exists(MODELS_DIR):
    # Remove existing models
    try:
        shutil.rmtree(MODELS_DIR)
        print("All existing models removed successfully.")
    except Exception as e:
        print(f"Error removing models: {str(e)}")

# Recreate models directory
os.makedirs(MODELS_DIR, exist_ok=True)

# Import embedding classes
print("Initializing embedding models...")
from embeddings import TfidfDocumentVectorizer, BERTDocumentVectorizer, Word2VecDocumentVectorizer
from search_engine import TextProcessor

# Path to the data directory
DATA_DIR = os.environ.get("DATA_DIR", "./data")

def initialize_embedding_model(model_type, dataset_name="imdb_sample"):
    """Initialize an embedding model and create placeholder files if needed."""
    print(f"Initializing {model_type} embedding model...")
    model_dir = os.path.join(MODELS_DIR, f"{dataset_name}_{model_type}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create sample texts for initialization
    dummy_texts = [
        "This movie is a masterpiece. The acting, direction, and cinematography are top-notch.",
        "I've never been so bored watching a film. Complete waste of time and money."
    ]
    
    # Initialize the appropriate vectorizer
    if model_type == "tfidf":
        vectorizer = TfidfDocumentVectorizer(min_df=1, max_df=1.0)
    elif model_type == "word2vec":
        vectorizer = Word2VecDocumentVectorizer()
    elif model_type == "bert":
        vectorizer = BERTDocumentVectorizer()
    else:
        raise ValueError(f"Unsupported embedding type: {model_type}")
    
    # Save the empty vectorizer to ensure directories exist
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    if not os.path.exists(vectorizer_path):
        # Force model download/initialization
        try:
            # Use multiple documents for better initialization
            vectors = vectorizer.fit_transform(dummy_texts)
            
            # Save the vectorizer
            vectorizer.save(vectorizer_path)
            print(f"Model {model_type} initialized and saved!")
            
            # Create placeholder files with better initialization
            empty_docs = dummy_texts
            empty_vectors = vectors
            empty_metadata = [
                {"id": 1, "raw_text": dummy_texts[0], "sentiment": "positive"},
                {"id": 2, "raw_text": dummy_texts[1], "sentiment": "negative"}
            ]
            
            with open(os.path.join(model_dir, "documents.pkl"), 'wb') as f:
                pickle.dump(empty_docs, f)
            
            with open(os.path.join(model_dir, "document_vectors.pkl"), 'wb') as f:
                pickle.dump(empty_vectors, f)
            
            with open(os.path.join(model_dir, "metadata.pkl"), 'wb') as f:
                pickle.dump(empty_metadata, f)
                
        except Exception as e:
            print(f"Error initializing {model_type} model: {str(e)}")

def main():
    start_time = time.time()
    print("Preparing embedding models...")
    
    # Check if data directory and sample dataset exist
    if os.path.exists(DATA_DIR):
        print(f"Data directory found at {DATA_DIR}")
        sample_path = os.path.join(DATA_DIR, "imdb_sample.csv")
        
        if os.path.exists(sample_path):
            print(f"Sample dataset found at {sample_path}")
        else:
            print("Sample dataset not found. It will be created when you run the search engine.")
    else:
        print(f"Data directory not found. Creating {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)
        
    # Initialize all embedding models
    for model_type in ["tfidf", "word2vec", "bert"]:
        initialize_embedding_model(model_type)
    
    elapsed_time = time.time() - start_time
    print(f"Model initialization completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main() 