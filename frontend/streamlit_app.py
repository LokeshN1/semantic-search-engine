import streamlit as st
import requests
import pandas as pd
import os
import sys
import time
from urllib.parse import urljoin

# Add backend directory to path for imports if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Base URL - use environment variable if available, with fallbacks
API_URL = os.environ.get("API_URL", "https://search-engine-api-gx3x.onrender.com")

# For local development
if API_URL.lower() == "local" or API_URL.lower() == "localhost":
    API_URL = "http://localhost:8000"

# Show what API URL we're using (temporarily for debugging)
api_debug = st.empty()

# Remove trailing slash if present to ensure urljoin works correctly
if API_URL.endswith('/'):
    API_URL = API_URL[:-1]

# ------------------ Sidebar Section ------------------
st.sidebar.title("🔍 Search Engine Controls")


def check_api_connection():
    try:
        response = requests.get(f"{API_URL}", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False

# API Connection Check
if not check_api_connection():
    st.sidebar.error("❌ Cannot connect to API server.")
    st.sidebar.info("Make sure to run the server:\n\ncd backend && uvicorn app:app --reload")
    st.stop()
else:
    # Create a persistent success message
    with st.sidebar:
        st.markdown("✅ *Connected to API Server!*")

# Fetch available datasets
def get_available_datasets():
    try:
        response = requests.get(urljoin(API_URL, "/datasets"))
        if response.status_code == 200:
            return response.json().get("datasets", {})
        else:
            return {}
    except:
        return {}

def refresh_datasets():
    try:
        response = requests.post(urljoin(API_URL, "/datasets/rescan"))
        if response.status_code == 200:
            st.sidebar.success("✅ Datasets refreshed successfully!")
            st.rerun()
        else:
            st.sidebar.error("❌ Failed to refresh datasets")
    except Exception as e:
        st.sidebar.error(f"❌ Error refreshing datasets: {e}")

def delete_dataset(dataset_name):
    try:
        response = requests.delete(urljoin(API_URL, f"/dataset/{dataset_name}"))
        if response.status_code == 200:
            st.sidebar.success(f"✅ Dataset '{dataset_name}' deleted successfully!")
            st.rerun()
        else:
            error_msg = response.json().get("detail", "Unknown error")
            st.sidebar.error(f"❌ Delete failed: {error_msg}")
    except Exception as e:
        st.sidebar.error(f"❌ Error deleting dataset: {e}")

# Dataset Section
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    st.markdown("### 🗂️ Datasets")
with col2:
    if st.button("🔄", help="Refresh dataset list"):
        refresh_datasets()

# Get the datasets
available_datasets = get_available_datasets()
dataset_options = list(available_datasets.keys())

if dataset_options:
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        dataset_options,
        format_func=lambda x: f"{x} ({available_datasets[x]['record_count']} records)"
    )
    
    # Check if the dataset was uploaded by the user
    is_user_uploaded = available_datasets[selected_dataset].get("user_uploaded", False)
    
    # Only show delete button for user-uploaded datasets
    if is_user_uploaded:
        if st.sidebar.button("🗑️ Delete Selected Dataset"):
            delete_dataset(selected_dataset)
    else:
        st.sidebar.info("Sample datasets are not deletable")
else:
    st.sidebar.warning("⚠️ No datasets available. Upload a dataset first.")
    selected_dataset = None

# Embedding model selection
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Search Options")
embedding_options = ["tfidf", "word2vec"]
embedding_help = "Choose the model to generate document embeddings. BERT is disabled on free tier."

# Try to connect to API to check if BERT is enabled
try:
    response = requests.get(f"{API_URL}", timeout=2)
    if response.status_code == 200:
        api_info = response.json()
        if api_info.get("bert_disabled") != True:  # API will add this field in future
            embedding_options.append("bert")
            embedding_help = "Choose the model to generate document embeddings."
except:
    pass  # Keep the default options if we can't connect

embedding_type = st.sidebar.selectbox(
    "Embedding Model",
    embedding_options,
    help=embedding_help
)

# Number of results
top_n = st.sidebar.slider(
    "Number of Results",
    min_value=1,
    max_value=5,
    value=3
)

# Upload dataset section
st.sidebar.markdown("---")
st.sidebar.subheader("⬆️ Upload New Dataset")

uploaded_file = st.sidebar.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])
dataset_name = st.sidebar.text_input("Dataset Name (alphanumeric)", "")

if uploaded_file and st.sidebar.button("Upload Dataset"):
    if not dataset_name.isalnum():
        st.sidebar.error("Dataset name must be alphanumeric only!")
    else:
        try:
            files = {"file": uploaded_file}
            data = {"dataset_name": dataset_name}

            with st.spinner("Uploading..."):
                upload_response = requests.post(urljoin(API_URL, "/upload"), files=files, data=data)

            if upload_response.status_code == 200:
                st.sidebar.success(f"✅ Dataset '{dataset_name}' uploaded successfully!")
                st.rerun()
            else:
                error_msg = upload_response.json().get("detail", "Unknown error")
                st.sidebar.error(f"❌ Upload failed: {error_msg}")

        except Exception as e:
            st.sidebar.error(f"Error uploading dataset: {e}")

# ------------------ Main Content Section ------------------
st.title("🔍 Semantic Search Engine")
st.markdown("Find documents based on **meaning**, not just keywords.")

query = st.text_input("Enter your search query", placeholder="e.g., 'action movies with strong female lead'")

if st.button("Search", disabled=not selected_dataset) and query:
    with st.spinner("Searching..."):
        try:
            start_time = time.time()

            params = {
                "query": query,
                "dataset": selected_dataset,
                "embedding": embedding_type,
                "top_n": top_n
            }
            response = requests.get(urljoin(API_URL, "/search"), params=params)

            if response.status_code == 200:
                result_data = response.json()
                search_results = result_data.get("results", [])
                exec_time = result_data.get("execution_time_ms", 0)

                st.success(f"🔎 Found {len(search_results)} results in {exec_time:.2f} ms!")

                if search_results:
                    for idx, result in enumerate(search_results):
                        similarity = result.get("similarity_score", 0)
                        text = result.get("text") or result.get("raw_text") or ""

                        if not text and result.get("metadata"):
                            for val in result["metadata"].values():
                                if isinstance(val, str) and len(val) > 20:
                                    text = val
                                    break

                        st.subheader(f"Result {idx+1} - Similarity: {similarity:.4f}")
                        st.write(text if text else "No text available.")

                        if "metadata" in result and result["metadata"]:
                            with st.expander("📄 Show Metadata"):
                                metadata_df = pd.DataFrame(
                                    [(k, v if len(str(v)) < 100 else str(v)[:100] + "...") 
                                     for k, v in result["metadata"].items()],
                                    columns=["Field", "Value"]
                                ).sort_values("Field")
                                st.table(metadata_df)

                        st.markdown("---")
                else:
                    st.info("🔔 No results found. Try rephrasing your query.")

            else:
                error_msg = response.json().get("detail", "Unknown error")
                st.error(f"❌ Search failed: {error_msg}")

        except Exception as e:
            st.error(f"❌ Search failed: {e}")
            st.info("Is your API server running?\n\nRun: cd backend && uvicorn app:app --reload")

# ------------------ About Section ------------------
st.markdown("---")
st.subheader("📚 About This App")
st.markdown("""
This semantic search engine finds documents based on their **meaning**, not just keyword matches.

**Key Features:**
- Use BERT, TF-IDF, or Word2Vec for embedding documents.
- Natural language querying.
- Ranked search results with similarity scores.
- Upload your own datasets (CSV or JSON).

**Example Queries:**
- "masterpiece with great acting"
- "emotional story with good cinematography"
- "The plot twists kept me engaged throughout the entire movie"

""")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("Semantic Search Engine | Built with Python, FastAPI, and Streamlit | [GitHub](https://github.com/lokeshN1/semantic-search-engine)")
