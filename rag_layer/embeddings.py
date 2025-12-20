"""Embedding generation for RAG using sentence transformers."""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate embeddings for text using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dimension}")
        
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of embedding vectors
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dimension
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
