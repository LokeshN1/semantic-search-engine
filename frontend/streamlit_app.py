import streamlit as st
import requests
import pandas as pd
import os
import sys
import time

# Add backend directory to path for imports if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Initialize session state
if 'upload_success' not in st.session_state:
    st.session_state.upload_success = False
if 'delete_success' not in st.session_state:
    st.session_state.delete_success = False

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One-time refresh handler
refresh_handler = """
<script>
    // Check if we need to reload the page only once
    document.addEventListener('DOMContentLoaded', function() {
        if (sessionStorage.getItem('refresh_required')) {
            // Clear the flag immediately to prevent multiple refreshes
            sessionStorage.removeItem('refresh_required');
        }
        
        // Auto-hide success messages after 5 seconds
        setTimeout(function() {
            // Find all success message elements and hide them
            var successElements = document.querySelectorAll('.stSuccess');
            successElements.forEach(function(element) {
                element.style.opacity = '0';
                setTimeout(function() { 
                    element.style.display = 'none'; 
                }, 1000);
            });
        }, 5000);  // 5 seconds
    });
</script>
"""
st.markdown(refresh_handler, unsafe_allow_html=True)

# API Base URL - use environment variable if available, with fallbacks
API_URL = os.environ.get("API_URL", "https://search-engine-api-gx3x.onrender.com")

# For local development
if API_URL.lower() == "local" or API_URL.lower() == "localhost":
    API_URL = "http://localhost:8000"

# Remove trailing slash if present
if API_URL.endswith('/'):
    API_URL = API_URL[:-1]

# Show what URL we're using (debug only)
api_debug = st.empty()

# ------------------ Sidebar Section ------------------
st.sidebar.title("🔍 Search Engine Controls")


def check_api_connection():
    global API_URL  # Declare the global variable at the beginning of the function
    
    # Hard-coded backend URL - use both the environment variable and explicit URL
    backends = [
        API_URL,  # Try environment variable first
        "https://search-engine-api-gx3x.onrender.com",  # Explicit URL as fallback
        "http://localhost:8000"  # Local development fallback
    ]
    
    for backend_url in backends:
        try:
            # st.toast(f"Trying to connect to {backend_url}...")
            response = requests.get(f"{backend_url}", timeout=5)
            if response.status_code == 200:
                # If successful, update the global API_URL to the working URL
                API_URL = backend_url
                return True
        except requests.RequestException as e:
            st.toast(f"Connection error: {str(e)[:50]}...")
            continue
    
    return False

# API Connection Check
connection_status = check_api_connection()
if not connection_status:
    st.sidebar.error("❌ Cannot connect to API server.")
    st.sidebar.info(f"Tried connecting to: {API_URL}")
    st.sidebar.info("Make sure both services are deployed correctly on Render")
    st.stop()
else:
    # Create a persistent success message
    with st.sidebar:
        st.markdown("✅ *Connected to API Server!*")

# Fetch available datasets
def get_available_datasets():
    try:
        endpoint = f"{API_URL}/datasets"
        response = requests.get(endpoint)
        if response.status_code == 200:
            return response.json().get("datasets", {})
        else:
            return {}
    except Exception as e:
        st.toast(f"Error fetching datasets: {str(e)[:50]}...")
        return {}



def delete_dataset(dataset_name):
    try:
        # Confirm we're deleting the right dataset
        endpoint = f"{API_URL}/dataset/{dataset_name}"
        
        # First check if this dataset exists and is user-uploaded
        check_endpoint = f"{API_URL}/dataset/{dataset_name}/is-user-uploaded"
        check_response = requests.get(check_endpoint)
        
        if check_response.status_code != 200 or not check_response.json().get("user_uploaded", False):
            st.sidebar.error(f"❌ Cannot delete dataset '{dataset_name}'. Only user-uploaded datasets can be deleted.")
            return
        
        # If it's confirmed to be a user-uploaded dataset, proceed with deletion
        response = requests.delete(endpoint)
        
        if response.status_code == 200:
            # Set delete success state instead of refreshing immediately
            st.session_state.delete_success = True
            st.session_state.deleted_dataset_name = dataset_name
            st.rerun()  # Just to show the success message
        else:
            error_msg = response.json().get("detail", "Unknown error")
            st.sidebar.error(f"❌ Delete failed: {error_msg}")
    except Exception as e:
        st.sidebar.error(f"❌ Error deleting dataset: {e}")

# Dataset Section
st.sidebar.markdown("---")
st.sidebar.subheader("🗂️ Datasets")

# Show delete success message if there was a recent deletion
if st.session_state.get('delete_success', False):
    st.sidebar.success(f"✅ Dataset '{st.session_state.deleted_dataset_name}' deleted successfully!")
    # Provide a manual refresh button
    if st.sidebar.button("🔄 Refresh Dataset List"):
        html_reload = """
        <script>
            window.location.href = window.location.href;
        </script>
        """
        st.markdown(html_reload, unsafe_allow_html=True)
    # Reset delete success state after displaying
    st.session_state.delete_success = False

# Get the datasets
available_datasets = get_available_datasets()
dataset_options = list(available_datasets.keys())



# Dataset selection
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
        delete_col1, delete_col2 = st.sidebar.columns([3, 2])
        with delete_col2:
            if st.button("🗑️ Delete", help=f"Delete {selected_dataset}"):
                delete_dataset(selected_dataset)
    else:
        st.sidebar.caption("ℹ️ Sample dataset (cannot be deleted)")
else:
    st.sidebar.warning("⚠️ No datasets available. Upload a dataset first.")
    selected_dataset = None

# Embedding model selection
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Search Options")

search_col1, search_col2 = st.sidebar.columns([3, 1])
with search_col1:
    embedding_type = st.selectbox(
        "Embedding Model",
        ["bert", "tfidf", "word2vec"],
        help="Choose the model to generate document embeddings."
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

# Show success message outside the form if there was a recent upload
if st.session_state.get('upload_success', False):
    st.sidebar.success(f"✅ Dataset '{st.session_state.last_uploaded_dataset}' uploaded successfully!")
    # Provide a manual refresh button outside the form
    if st.sidebar.button("🔄 Refresh Dataset List"):
        html_reload = """
        <script>
            window.location.href = window.location.href;
        </script>
        """
        st.markdown(html_reload, unsafe_allow_html=True)
    # Reset success state after displaying
    st.session_state.upload_success = False

# Always show the upload form
with st.sidebar.form("upload_form"):
    uploaded_file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    dataset_name = st.text_input("Dataset Name (alphanumeric)")
    submit_button = st.form_submit_button("Upload Dataset")
    
    if submit_button:
        if not uploaded_file:
            st.error("❌ Please select a file to upload")
        elif not dataset_name:
            st.error("❌ Please enter a dataset name")
        elif not dataset_name.isalnum():
            st.error("❌ Dataset name must be alphanumeric only!")
        else:
            try:
                files = {"file": uploaded_file}
                data = {"dataset_name": dataset_name}
                
                endpoint = f"{API_URL}/upload"

                with st.spinner("Uploading..."):
                    upload_response = requests.post(endpoint, files=files, data=data)

                if upload_response.status_code == 200:
                    # Set success state and refresh to show success message
                    st.session_state.upload_success = True
                    st.session_state.last_uploaded_dataset = dataset_name
                    st.rerun()  # Just to show success message
                else:
                    error_msg = upload_response.json().get("detail", "Unknown error")
                    st.error(f"❌ Upload failed: {error_msg}")
            except Exception as e:
                st.error(f"Error uploading dataset: {e}")

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
            
            endpoint = f"{API_URL}/search"
            response = requests.get(endpoint, params=params)

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
