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
    A simplified document vectorizer using TF-IDF as a fallback when sentence-transformers is not available.
    This is a placeholder that uses TF-IDF instead of actual BERT embeddings to avoid dependency issues.
    """
    
    def __init__(self, max_features=768):
        """
        Initialize the BERT document vectorizer.
        
        Args:
            max_features: Number of features to match BERT embedding dimensions
        """
        self.vector_size = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2)  # Use unigrams and bigrams to capture more context
        )
        self.is_fitted = False
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit the vectorizer on the documents.
        
        Args:
            documents: List of preprocessed document texts
        """
        self.vectorizer.fit(documents)
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
        
        # Transform to TF-IDF vectors
        vectors = self.vectorizer.transform(documents).toarray()
        
        # If we have fewer features than expected, pad with zeros
        if vectors.shape[1] < self.vector_size:
            padding = np.zeros((vectors.shape[0], self.vector_size - vectors.shape[1]))
            vectors = np.hstack((vectors, padding))
            
        # If we have more features than expected, truncate
        elif vectors.shape[1] > self.vector_size:
            vectors = vectors[:, :self.vector_size]
            
        return vectors
    
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
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
    def load(self, path: str) -> None:
        """
        Load the vectorizer from disk.
        
        Args:
            path: Path to load the vectorizer from
        """
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True 