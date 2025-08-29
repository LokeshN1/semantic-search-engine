import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import pickle
from abc import ABC, abstractmethod
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Data preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextProcessor:
    """Class for text preprocessing tasks."""
    
    def __init__(self, use_stemming=False, use_lemmatization=True):
        # Download necessary NLTK data if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        if use_lemmatization:
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        
        if use_stemming:
            self.stemmer = PorterStemmer()
        if use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess the text by:
        1. Converting to lowercase
        2. Removing special characters and numbers
        3. Removing stopwords
        4. Stemming or lemmatization (if enabled)
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Apply stemming or lemmatization
        if self.use_stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]
        elif self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back into a string
        return ' '.join(tokens)


class Vectorizer(ABC):
    """Abstract base class for vectorization methods."""
    
    @abstractmethod
    def fit(self, documents: List[str]) -> None:
        """Fit the vectorizer on the documents."""
        pass
    
    @abstractmethod
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents into vector representations."""
        pass
    
    @abstractmethod
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit the vectorizer and transform documents."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the document vectors."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vectorizer to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vectorizer from disk."""
        pass


class SearchEngine:
    """Main search engine class that integrates preprocessing, vectorization, and search."""
    
    def __init__(self, 
                 vectorizer: Vectorizer,
                 text_processor: Optional[TextProcessor] = None,
                 cache_size: int = 100):
        self.vectorizer = vectorizer
        self.text_processor = text_processor if text_processor else TextProcessor()
        self.documents = []
        self.document_vectors = None
        self.metadata = []
        self.cache = {}  # Simple cache for query results
        self.cache_size = cache_size
    
    def load_data(self, data_path: str, text_column: str, metadata_columns: List[str] = None) -> None:
        """
        Load and preprocess documents from a CSV or JSON file.
        
        Args:
            data_path: Path to the dataset file
            text_column: Name of the column containing the document text
            metadata_columns: List of column names to include as metadata
        """
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # Extract text documents
        raw_documents = df[text_column].tolist()
        
        # Preprocess documents
        self.documents = [self.text_processor.preprocess(doc) for doc in raw_documents]
        
        # Extract metadata if specified
        if metadata_columns:
            self.metadata = df[metadata_columns].to_dict('records')
            
            # Add raw text to metadata
            for i, doc in enumerate(self.metadata):
                if i < len(raw_documents):
                    self.metadata[i]['raw_text'] = raw_documents[i]
        else:
            # Always include raw text and check for sentiment in metadata
            self.metadata = []
            for i, raw_doc in enumerate(raw_documents):
                metadata_item = {"id": i, "raw_text": raw_doc}
                
                # If 'sentiment' column exists in the dataframe, include it in metadata
                if 'sentiment' in df.columns:
                    metadata_item['sentiment'] = df['sentiment'].iloc[i]
                
                self.metadata.append(metadata_item)
    
    def build_index(self) -> None:
        """Build the search index by vectorizing documents."""
        self.document_vectors = self.vectorizer.fit_transform(self.documents)
    
    def search(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents semantically similar to the query.
        
        Args:
            query: The search query
            top_n: Number of top results to return
            
        Returns:
            List of dictionaries with document metadata and similarity scores
        """
        # Check if query is in cache
        if query in self.cache:
            return self.cache[query][:top_n]  # Return only top_n results from cache
        
        # Preprocess the query
        processed_query = self.text_processor.preprocess(query)
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity between query and all documents
        similarities = self._calculate_similarity(query_vector, self.document_vectors)
        
        # Get the indices of top_n most similar documents
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        # Create result list with metadata and similarity scores
        results = []
        for idx in top_indices:
            result = self.metadata[idx].copy()
            result["similarity_score"] = float(similarities[idx])
            results.append(result)
        
        # Update cache
        self._update_cache(query, results, top_n)
        
        return results[:top_n]  # Ensure only top_n results are returned
    
    def _calculate_similarity(self, query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query vector and document vectors.
        
        Args:
            query_vector: Vector representation of the query
            document_vectors: Matrix of document vectors
            
        Returns:
            Array of similarity scores
        """
        # Normalize vectors to unit length
        query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        if query_norm[0, 0] == 0:
            query_norm[0, 0] = 1  # Avoid division by zero
        
        query_vector_normalized = query_vector / query_norm
        
        # Normalize each document vector
        doc_norms = np.linalg.norm(document_vectors, axis=1, keepdims=True)
        # Replace zero norms with 1 to avoid division by zero
        doc_norms[doc_norms == 0] = 1
        document_vectors_normalized = document_vectors / doc_norms
        
        # Calculate cosine similarity (dot product of normalized vectors)
        # This will be between -1 and 1, where 1 means most similar
        similarities = np.dot(document_vectors_normalized, query_vector_normalized.T).flatten()
        
        # Scale to 0-1 range by adding 1 and dividing by 2
        similarities = (similarities + 1) / 2
        
        return similarities
    
    def _update_cache(self, query: str, results: List[Dict[str, Any]], top_n: int) -> None:
        """Update the query cache with new results."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest query if cache is full
            oldest_query = next(iter(self.cache))
            del self.cache[oldest_query]
        
        # Store only the top_n results in the cache
        self.cache[query] = results[:top_n]
        
    def save_model(self, model_dir: str) -> None:
        """Save the search engine model to disk."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save vectorizer
        self.vectorizer.save(os.path.join(model_dir, "vectorizer.pkl"))
        
        # Save document vectors
        with open(os.path.join(model_dir, "document_vectors.pkl"), 'wb') as f:
            pickle.dump(self.document_vectors, f)
        
        # Save metadata
        with open(os.path.join(model_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save documents
        with open(os.path.join(model_dir, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
    
    def load_model(self, model_dir: str) -> None:
        """Load the search engine model from disk."""
        # Load vectorizer
        self.vectorizer.load(os.path.join(model_dir, "vectorizer.pkl"))
        
        # Load document vectors
        with open(os.path.join(model_dir, "document_vectors.pkl"), 'rb') as f:
            self.document_vectors = pickle.load(f)
        
        # Load metadata
        with open(os.path.join(model_dir, "metadata.pkl"), 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load documents
        with open(os.path.join(model_dir, "documents.pkl"), 'rb') as f:
            self.documents = pickle.load(f) 