import numpy as np
import pickle
from typing import List
import os
import sys
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Add parent directory to path to find search_engine module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_engine import Vectorizer


class Word2VecDocumentVectorizer(Vectorizer):
    """
    Word2Vec based document vectorizer implementation using gensim.
    This creates semantically meaningful embeddings based on word context.
    """
    
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        """
        Initialize the Word2Vec document vectorizer.
        
        Args:
            vector_size: Dimensionality of the word vectors
            window: Maximum distance between the current and predicted word
            min_count: Ignores all words with total frequency lower than this
            workers: Number of worker threads to train the model
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.is_fitted = False
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit the Word2Vec model on the documents.
        
        Args:
            documents: List of preprocessed document texts
        """
        # Tokenize documents for Word2Vec training
        tokenized_docs = [word_tokenize(doc) for doc in documents]
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        
        self.is_fitted = True
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents into vector representations by averaging word vectors.
        
        Args:
            documents: List of preprocessed document texts
            
        Returns:
            Document vectors as a numpy array
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        document_vectors = np.zeros((len(documents), self.vector_size))
        
        for i, doc in enumerate(documents):
            words = word_tokenize(doc)
            word_vectors = []
            
            for word in words:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])
            
            if word_vectors:
                # Average the word vectors to get document vector
                document_vectors[i] = np.mean(word_vectors, axis=0)
        
        return document_vectors
    
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
        # Save model parameters
        params = {
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'workers': self.workers
        }
        
        # Create directory for Word2Vec model
        model_dir = os.path.dirname(path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save parameters
        with open(path, 'wb') as f:
            pickle.dump(params, f)
        
        # Save Word2Vec model
        if self.model is not None:
            model_path = os.path.join(model_dir, "word2vec_model")
            self.model.save(model_path)
        
    def load(self, path: str) -> None:
        """
        Load the vectorizer from disk.
        
        Args:
            path: Path to load the vectorizer from
        """
        # Load parameters
        with open(path, 'rb') as f:
            params = pickle.load(f)
        
        self.vector_size = params['vector_size']
        self.window = params['window']
        self.min_count = params['min_count']
        self.workers = params['workers']
        
        # Load Word2Vec model
        model_dir = os.path.dirname(path)
        model_path = os.path.join(model_dir, "word2vec_model")
        
        if os.path.exists(model_path):
            self.model = Word2Vec.load(model_path)
            self.is_fitted = True 