"""Vector store implementation using ChromaDB."""

import os
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np

from .document_loader import Document


class VectorStore:
    """Vector store for storing and querying document embeddings."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None
    ):
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document embeddings for RAG"}
        )
        
        print(f"Vector store initialized: {collection_name}")
        print(f"Current document count: {self.collection.count()}")
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None
    ):
        """Add documents and their embeddings to the store.
        
        Args:
            documents: List of Document objects
            embeddings: Array of embedding vectors
            ids: Optional list of document IDs
        """
        if ids is None:
            # Generate IDs based on collection count
            start_id = self.collection.count()
            ids = [f"doc_{start_id + i}" for i in range(len(documents))]
        
        # Prepare data for ChromaDB
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Query the vector store for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Tuple of (documents, metadatas, distances)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata
        )
        
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        return documents, metadatas, distances
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
    
    def get_count(self) -> int:
        """Get the number of documents in the store.
        
        Returns:
            Number of documents
        """
        return self.collection.count()
    
    def clear(self):
        """Clear all documents from the collection."""
        # Delete and recreate the collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document embeddings for RAG"}
        )
        print(f"Cleared collection: {self.collection_name}")
