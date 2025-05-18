import numpy as np
import pickle
from typing import List
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_engine import Vectorizer


class BERTDocumentVectorizer(Vectorizer):
    """
    Document vectorizer using sentence-transformers BERT model for semantic embeddings.
    Optimized for memory usage with model pooling and batch processing.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', max_length=512):
        """
        Initialize the BERT document vectorizer.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            max_length: Maximum sequence length for the model
        """
        self.vector_size = 384  # Default for all-MiniLM-L6-v2
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.is_fitted = False
        
        # Fall back to TF-IDF if sentence-transformers isn't available
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.vector_size,
            ngram_range=(1, 2)  # Use unigrams and bigrams to capture more context
        )
    
    def _load_model(self):
        """Load the sentence-transformers model if not already loaded."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Set environment variables to reduce memory usage
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                
                # Use the smaller model for embeddings
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                print("WARNING: sentence-transformers not available, using TF-IDF as fallback")
                self.model = None
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit the vectorizer on the documents.
        
        Args:
            documents: List of preprocessed document texts
        """
        try:
            self._load_model()
            # BERT models don't need fitting
            self.is_fitted = True
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            # Fall back to TF-IDF
            self.tfidf_vectorizer.fit(documents)
            self.is_fitted = True
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents into vector representations.
        
        Args:
            documents: List of preprocessed document texts
            
        Returns:
            Document vectors as a numpy array
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        try:
            self._load_model()
            if self.model:
                # Process in smaller batches to reduce memory usage
                batch_size = 32
                vectors = []
                
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i+batch_size]
                    batch_vectors = self.model.encode(
                        batch, 
                        show_progress_bar=False, 
                        convert_to_numpy=True,
                        max_length=self.max_length
                    )
                    vectors.append(batch_vectors)
                
                return np.vstack(vectors) if vectors else np.array([])
            else:
                raise ValueError("Model not available")
        except Exception as e:
            print(f"Error using BERT model: {e}, falling back to TF-IDF")
            # Fall back to TF-IDF
            return self.tfidf_vectorizer.transform(documents).toarray()
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit the vectorizer and transform documents.
        
        Args:
            documents: List of preprocessed document texts
            
        Returns:
            Document vectors as a numpy array
        """
        self.fit(documents)
        return self.transform(documents)
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the document vectors.
        
        Returns:
            The dimension of the document vectors
        """
        return self.vector_size
    
    def save(self, path: str) -> None:
        """
        Save the vectorizer to disk.
        
        Args:
            path: Path to save the vectorizer
        """
        # We only need to save the TF-IDF vectorizer as fallback
        with open(path, 'wb') as f:
            pickle.dump({
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'model_name': self.model_name,
                'max_length': self.max_length
            }, f)
        
    def load(self, path: str) -> None:
        """
        Load the vectorizer from disk.
        
        Args:
            path: Path to load the vectorizer from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                self.tfidf_vectorizer = data.get('tfidf_vectorizer', self.tfidf_vectorizer)
                self.model_name = data.get('model_name', self.model_name)
                self.max_length = data.get('max_length', self.max_length)
            else:
                # Backward compatibility with old format
                self.tfidf_vectorizer = data
        self.is_fitted = True 