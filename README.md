# Semantic Search Engine

A powerful semantic search engine using various embedding techniques (BERT, TF-IDF, Word2Vec) to search through document collections based on meaning rather than keywords.

## Features

- **Multiple Embedding Options**: Choose between BERT, TF-IDF, and Word2Vec embeddings
- **Document Processing**: Automatic text preprocessing and metadata handling
- **REST API**: FastAPI-powered endpoint for search integration
- **Web Interface**: Streamlit-based UI for easy searching and result viewing
- **Model Caching**: Save and load pre-trained models for faster startup

## Deployment

This project is configured for deployment on Render with a separate backend API service and frontend Streamlit application.

### How to Deploy

1. **Sign up for [Render](https://render.com)** if you don't already have an account.

2. **Create a new "Blueprint" instance** on Render, and connect to your GitHub repository.

3. **Deploy the Blueprint**
   - Render will automatically detect the `render.yaml` file and set up both services.
   - The backend API will be deployed at `https://search-engine-api.onrender.com`
   - The frontend will be deployed at `https://search-engine-frontend.onrender.com`

### Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Initialize resources: `python initialize_deployment.py`
3. Run backend: `cd backend && uvicorn app:app --reload`
4. Run frontend: `cd frontend && streamlit run streamlit_app.py`

