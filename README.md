# Semantic Search Engine

A powerful semantic search engine using various embedding techniques (BERT, TF-IDF, Word2Vec) to search through document collections based on meaning rather than keywords.

### Architechture
![Architecture](/data/images/SemanticSearchEngineArchiteture.png)

## Screenshots

### Main Interface
![Main Interface](/data/images/main_interface.png)

### Upload Dataset Interface
![Upload Dataset](/data/images/upload_dataset.png)

### Search Options
![Search Options](/data/images/search_options.png)


### Multiple Results View
![Multiple Results](/data/images/multiple_results.png)

### Search Results with Metadata
![Search Results with Metadata](/data/images/search_results_metadata.png)



## Features

- **Multiple Embedding Options**: Choose between BERT, TF-IDF, and Word2Vec embeddings
- **Document Processing**: Automatic text preprocessing and metadata handling
- **REST API**: FastAPI-powered endpoint for search integration
- **Web Interface**: Streamlit-based UI for easy searching and result viewing
- **Model Caching**: Save and load pre-trained models for faster startup


## Dataset Requirements

The search engine supports both CSV and JSON datasets with the following requirements:

- **CSV Format**:
  - Must contain at least one text column with one of these names:
    - `text`
    - `content`
    - `description`
    - `title`
    - `review`
  - If none of these column names are found, the first non-ID column will be used
  - Optional metadata columns

- **JSON Format**:
  - Array of objects with at least one text field
  - Each object can have additional metadata fields




### Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Initialize resources: `python initialize_deployment.py`
3. Run backend: `cd backend && uvicorn app:app --reload`
4. Run frontend: `cd frontend && streamlit run streamlit_app.py`






