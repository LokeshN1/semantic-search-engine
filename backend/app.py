import os
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import uvicorn
import time
import traceback
import shutil

from search_engine import SearchEngine, TextProcessor
from embeddings import TfidfDocumentVectorizer, BERTDocumentVectorizer, Word2VecDocumentVectorizer

# Define models for requests and responses
class SearchQuery(BaseModel):
    query: str
    top_n: int = 5
    embedding_type: str = "bert"

class SearchResult(BaseModel):
    id: int
    similarity_score: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    execution_time_ms: float
    embedding_type: str

# Create FastAPI application
app = FastAPI(
    title="Semantic Search Engine API",
    description="API for semantic search over document collections",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "https://search-engine-frontend.onrender.com", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Global variables to store search engines
search_engines = {}
available_datasets = {}
user_uploaded_datasets = set()  # Track which datasets were uploaded by users
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))
MODEL_CACHE_TIME = int(os.environ.get("MODEL_CACHE_TIME", 3600))  # Default: 1 hour

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Function to initialize search engine with selected embedding method
def get_search_engine(dataset_name: str, embedding_type: str = "bert", force_rebuild: bool = False) -> SearchEngine:
    """Get or create a search engine for the specified dataset and embedding type."""
    cache_key = f"{dataset_name}_{embedding_type}"
    
    if cache_key in search_engines and not force_rebuild:
        return search_engines[cache_key]
    
    # Create the vectorizer based on the embedding type
    if embedding_type == "tfidf":
        vectorizer = TfidfDocumentVectorizer(min_df=1, max_df=1.0)
    elif embedding_type == "word2vec":
        vectorizer = Word2VecDocumentVectorizer()
    elif embedding_type == "bert":
        vectorizer = BERTDocumentVectorizer()
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    # Create text processor
    text_processor = TextProcessor()
    
    # Create search engine
    search_engine = SearchEngine(vectorizer=vectorizer, text_processor=text_processor)
    
    # Check if a pre-trained model exists
    model_dir = os.path.join(MODELS_DIR, cache_key)
    data_file = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    
    # If force_rebuild is True or the data file doesn't match what we have cached, force rebuilding the index
    if force_rebuild and os.path.exists(model_dir):
        print(f"Forcing rebuild of {cache_key}")
        shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(model_dir) and os.path.isdir(model_dir) and not force_rebuild:
        # Load pre-existing model
        try:
            search_engine.load_model(model_dir)
            search_engines[cache_key] = search_engine
            return search_engine
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # If loading fails, we'll rebuild the model
    
    # Load and process data
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    
    # Determine the text column and metadata columns
    df = pd.read_csv(data_file)
    text_column = next((col for col in df.columns if col.lower() in ["text", "content", "description", "title", "review"]), df.columns[1])
    id_column = next((col for col in df.columns if col.lower() in ["id", "index"]), df.columns[0])
    
    # Include ALL columns as metadata (except text column)
    metadata_columns = [col for col in df.columns if col != text_column]
    print(f"Using metadata columns: {metadata_columns}")
    
    # Load data and build index
    print(f"Building search index for {dataset_name} with {embedding_type} embeddings...")
    search_engine.load_data(data_file, text_column=text_column, metadata_columns=metadata_columns)
    search_engine.build_index()
    
    # Save the model
    os.makedirs(model_dir, exist_ok=True)
    search_engine.save_model(model_dir)
    
    # Cache the search engine
    search_engines[cache_key] = search_engine
    
    return search_engine

# Scan for available datasets
def scan_datasets():
    """Scan for datasets in the data directory."""
    global available_datasets
    
    available_datasets = {}
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv') or file.endswith('.json'):
            dataset_name = os.path.splitext(file)[0]
            file_path = os.path.join(DATA_DIR, file)
            
            try:
                df = pd.read_csv(file_path) if file.endswith('.csv') else pd.read_json(file_path)
                record_count = len(df)
                available_datasets[dataset_name] = {
                    "file": file,
                    "record_count": record_count,
                    "columns": list(df.columns)
                }
                
                # Force rebuild models for this dataset to ensure we use the actual data
                try:
                    for embedding_type in ["tfidf", "word2vec", "bert"]:
                        get_search_engine(dataset_name, embedding_type, force_rebuild=True)
                except Exception as e:
                    print(f"Error initializing search engines for {dataset_name}: {str(e)}")
                    
            except Exception as e:
                print(f"Error reading dataset {file}: {str(e)}")
    
    return available_datasets

# API routes
@app.get("/")
async def root():
    """API root endpoint with information about the API."""
    return {
        "name": "Semantic Search Engine API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": [
            {"path": "/search", "method": "GET", "description": "Search documents"},
            {"path": "/datasets", "method": "GET", "description": "List available datasets"},
            {"path": "/datasets/rescan", "method": "POST", "description": "Force rescan of dataset directory"},
            {"path": "/upload", "method": "POST", "description": "Upload new dataset"},
            {"path": "/dataset/{dataset_name}", "method": "DELETE", "description": "Delete a dataset"},
            {"path": "/dataset/{dataset_name}/is-user-uploaded", "method": "GET", "description": "Check if dataset was uploaded by user"}
        ]
    }

@app.get("/search")
async def search(
    query: str = Query(..., description="Search query"),
    dataset: str = Query("default", description="Dataset name to search in"),
    embedding: str = Query("bert", description="Embedding type (tfidf, word2vec, bert)"),
    top_n: int = Query(5, description="Number of results to return", gt=0, le=100)
):
    """
    Search documents using the semantic search engine.
    """
    if dataset not in available_datasets and dataset != "default":
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")
    
    # Use default dataset if none specified
    if dataset == "default":
        available = list(available_datasets.keys())
        if not available:
            raise HTTPException(status_code=404, detail="No datasets available")
        dataset = available[0]
    
    try:
        start_time = time.time()
        
        # Get search engine
        search_engine = get_search_engine(dataset, embedding)
        
        # Perform search
        search_results = search_engine.search(query, top_n=top_n)
        
        # Print results for debugging (with limited output to avoid flooding logs)
        print(f"Raw search results: {[{k: (v[:30] + '...' if isinstance(v, str) and len(v) > 30 else v) for k, v in r.items()} for r in search_results[:2]]}")
        
        # Format results
        formatted_results = []
        for i, result in enumerate(search_results):
            # Create a copy of the result to avoid modifying the original
            result_item = {
                "id": result.get("id", i),
                "similarity_score": result["similarity_score"]
            }
            
            # Always include raw_text if it exists
            if "raw_text" in result:
                result_item["raw_text"] = result["raw_text"]
            
            # Explicitly look for the review text in different possible fields
            text_found = False
            
            # First check for the 'review' column (specific to IMDB dataset)
            if "review" in result:
                result_item["text"] = result["review"]
                text_found = True
            # Then check for generic text fields
            elif "raw_text" in result:
                result_item["text"] = result["raw_text"]
                text_found = True
            elif "text" in result:
                result_item["text"] = result["text"]
                text_found = True
            # If no text field found, use the first string field that's not id or sentiment
            else:
                for key, value in result.items():
                    if isinstance(value, str) and key not in ["id", "sentiment"] and len(value) > 10:
                        result_item["text"] = value
                        text_found = True
                        break
            
            # If still no text found, add a placeholder
            if not text_found:
                result_item["text"] = "No text content available for this result."
            
            # Add all other metadata
            metadata = {}
            for key, value in result.items():
                if key not in ["id", "similarity_score", "raw_text"] and key != "text":
                    # Ensure all values are properly serializable
                    if isinstance(value, (int, float, bool, str)):
                        metadata[key] = value
                    else:
                        # Convert other types to string to ensure they can be serialized
                        try:
                            metadata[key] = str(value)
                        except:
                            # Skip values that can't be serialized
                            continue
            
            if metadata:
                result_item["metadata"] = metadata
                
            formatted_results.append(result_item)
        
        # Print a sample of formatted results for debugging
        if formatted_results:
            print(f"Sample formatted result: {formatted_results[0]}")
        
        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Create response
        response = {
            "results": formatted_results,
            "query": query,
            "execution_time_ms": execution_time,
            "embedding_type": embedding
        }
        
        return response
        
    except Exception as e:
        print(f"Error in search: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets")
async def list_datasets():
    """List all available datasets."""
    # Add a flag to indicate which datasets were uploaded by users
    datasets_with_flags = {}
    for name, data in available_datasets.items():
        datasets_with_flags[name] = {**data, "user_uploaded": name in user_uploaded_datasets}
    
    return {"datasets": datasets_with_flags}

@app.get("/dataset/{dataset_name}/is-user-uploaded")
async def is_user_uploaded(dataset_name: str):
    """Check if a dataset was uploaded by the user."""
    if dataset_name not in available_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    return {"dataset": dataset_name, "user_uploaded": dataset_name in user_uploaded_datasets}

@app.post("/datasets/rescan")
async def rescan_datasets():
    """Force rescan of the datasets directory."""
    try:
        scan_datasets()
        return {"message": "Datasets rescanned successfully", "datasets": available_datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(..., description="Name for the dataset"),
    text_column: Optional[str] = Form(None, description="Column containing document text")
):
    """Upload a new dataset."""
    try:
        # Validate dataset name (alphanumeric only)
        if not dataset_name.isalnum():
            raise HTTPException(status_code=400, detail="Dataset name must be alphanumeric")
        
        # Save the file
        file_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Add to user uploaded datasets
        user_uploaded_datasets.add(dataset_name)
        
        # Rescan datasets
        scan_datasets()
        
        return {"message": f"Dataset '{dataset_name}' uploaded successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/dataset/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset by name."""
    try:
        if dataset_name not in available_datasets:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        # Check if dataset was uploaded by user
        if dataset_name not in user_uploaded_datasets:
            raise HTTPException(status_code=403, detail=f"Cannot delete dataset '{dataset_name}'. Only user-uploaded datasets can be deleted.")
        
        # Get the file path from the available_datasets dict
        file_name = available_datasets[dataset_name]["file"]
        file_path = os.path.join(DATA_DIR, file_name)
        
        # Remove the file
        os.remove(file_path)
        
        # Also remove any associated model directories
        for embedding_type in ["tfidf", "word2vec", "bert"]:
            model_dir = os.path.join(MODELS_DIR, f"{dataset_name}_{embedding_type}")
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
        
        # Remove from search engines cache
        for embedding_type in ["tfidf", "word2vec", "bert"]:
            cache_key = f"{dataset_name}_{embedding_type}"
            if cache_key in search_engines:
                del search_engines[cache_key]
        
        # Remove from user uploaded datasets
        user_uploaded_datasets.remove(dataset_name)
        
        # Rescan datasets
        scan_datasets()
        
        return {"message": f"Dataset '{dataset_name}' deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    try:
        print("Starting application initialization...")
        print(f"DATA_DIR: {DATA_DIR}")
        print(f"MODELS_DIR: {MODELS_DIR}")
        
        # Perform dataset scanning
        global available_datasets
        available_datasets = scan_datasets()
        print(f"Found {len(available_datasets)} datasets: {list(available_datasets.keys())}")
        
        # Load default dataset if available
        if available_datasets:
            default_dataset = list(available_datasets.keys())[0]
            print(f"Preloading default dataset: {default_dataset}")
            # Don't force rebuild during startup to make it faster
            for embedding_type in ["tfidf", "bert"]:
                try:
                    get_search_engine(default_dataset, embedding_type, force_rebuild=False)
                    print(f"Successfully loaded {embedding_type} model for {default_dataset}")
                except Exception as e:
                    print(f"Error loading {embedding_type} model: {str(e)}")
                    traceback.print_exc()
        print("Application initialization completed successfully!")
    except Exception as e:
        print(f"Critical error during startup: {str(e)}")
        traceback.print_exc()
        # Continue anyway to allow manual troubleshooting

# Add a debug endpoint
@app.get("/debug")
async def debug():
    """Debug endpoint to test API connectivity."""
    return {
        "status": "ok",
        "message": "API connection successful",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)