# Semantic Search Engine

A powerful semantic search engine using various embedding techniques (BERT, TF-IDF, Word2Vec) to search through document collections based on meaning rather than keywords.

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



## Deployment

This project is configured for deployment on Render with a separate backend API service and frontend Streamlit application.

### How to Deploy

1. **Sign up for [Render](https://render.com)** if you don't already have an account.

2. **Create a new "Blueprint" instance** on Render, and connect to your GitHub repository.

3. **Deploy the Blueprint**
   - Render will automatically detect the `render.yaml` file and set up both services.

### Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Initialize resources: `python initialize_deployment.py`
3. Run backend: `cd backend && uvicorn app:app --reload`
4. Run frontend: `cd frontend && streamlit run streamlit_app.py`

### Important Note About Free Instance

The search engine for this project is hosted on Render's free tier, which comes with a few constraints:


- **Sleep Mode**: Free instances spin down after 15 minutes of inactivity
- **Cold Start**: First request after sleep mode may take 30-60 seconds to respond
- **Connection Issues**: You might occasionally see "Cannot connect to API server" error
  - This is normal and usually resolves itself within a minute
  - Simply refresh the page if you encounter this error
  - For production use, consider upgrading to a paid instance

To minimize these issues:
1. Wait a few seconds after the first load
2. Refresh the page if you see connection errors
3. Consider running the application locally for development




