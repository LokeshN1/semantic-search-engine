import numpy as np
import pickle
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os

# Add parent directory to path to find search_engine module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_engine import Vectorizer


class TfidfDocumentVectorizer(Vectorizer):
    """TF-IDF based document vectorizer implementation."""
    
    def __init__(self, max_features=5000, min_df=1, max_df=1.0):
        """
        Initialize the TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features to consider
            min_df: Minimum document frequency threshold
            max_df: Maximum document frequency threshold
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df
        )
        self.is_fitted = False
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit the vectorizer on the documents.
        
        Args:
            documents: List of preprocessed document texts
        """
        # For small datasets, ensure we don't have incompatible min_df and max_df
        if len(documents) <= 2:
            self.vectorizer.min_df = 1
            self.vectorizer.max_df = 1.0
            
        self.vectorizer.fit(documents)
        self.is_fitted = True
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents into TF-IDF vector representations.
        
        Args:
            documents: List of preprocessed document texts
            
        Returns:
            Document vectors as a numpy array
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        return self.vectorizer.transform(documents).toarray()
    
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
        if not self.is_fitted:
            # If not fitted, return the max_features or a default
            return getattr(self.vectorizer, 'max_features', 5000)
        
        # Return the actual vocabulary size
        return len(self.vectorizer.vocabulary_)
    
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