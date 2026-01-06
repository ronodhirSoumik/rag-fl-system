"""Integrated Federated Learning and RAG System."""

from typing import List, Optional
import os

from rag_layer import (
    DocumentLoader,
    EmbeddingGenerator,
    VectorStore,
    Retriever,
    load_documents
)
from .config import Config, get_config


class FederatedRAGSystem:
    """Combined system integrating Federated Learning and RAG.
    
    This system allows for:
    1. Privacy-preserving federated learning across distributed clients
    2. Knowledge retrieval from a centralized document store
    3. Enhanced model training with retrieved context
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the Federated RAG System.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or get_config()
        
        print("Initializing Federated RAG System...")
        
        # Initialize RAG components
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config.rag_embedding_model
        )
        
        self.vector_store = VectorStore(
            collection_name=self.config.rag_collection_name,
            persist_directory=self.config.vector_store_path
        )
        
        self.retriever = Retriever(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store
        )
        
        self.document_loader = DocumentLoader(
            chunk_size=self.config.rag_chunk_size,
            chunk_overlap=self.config.rag_chunk_overlap
        )
        
        print("Federated RAG System initialized successfully")
    
    def load_knowledge_base(self, source_path: str, is_directory: bool = False):
        """Load documents into the RAG knowledge base.
        
        Args:
            source_path: Path to document(s)
            is_directory: Whether source_path is a directory
        """
        print(f"\nLoading knowledge base from: {source_path}")
        
        documents = load_documents(
            source=source_path,
            chunk_size=self.config.rag_chunk_size,
            chunk_overlap=self.config.rag_chunk_overlap,
            is_directory=is_directory
        )
        
        print(f"Loaded {len(documents)} document chunks")
        
        if documents:
            self.retriever.add_documents(documents)
            print("Knowledge base loaded successfully")
        else:
            print("Warning: No documents were loaded")
    
    def query_knowledge_base(self, query: str, top_k: Optional[int] = None) -> str:
        """Query the RAG knowledge base.
        
        Args:
            query: Query string
            top_k: Number of results to return (uses config default if None)
            
        Returns:
            Formatted context string
        """
        top_k = top_k or self.config.rag_top_k
        
        print(f"\nQuerying knowledge base: '{query}'")
        print(f"Retrieving top {top_k} results...")
        
        results = self.retriever.retrieve(query, top_k=top_k)
        
        if not results:
            return "No relevant information found."
        
        print(f"\nFound {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.4f} | {result.content[:80]}...")
        
        return self.retriever.get_context(query, top_k=top_k)
    
    def get_retrieval_results(self, query: str, top_k: Optional[int] = None):
        """Get detailed retrieval results.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        top_k = top_k or self.config.rag_top_k
        return self.retriever.retrieve(query, top_k=top_k)
    
    def get_stats(self) -> dict:
        """Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "total_documents": self.retriever.get_document_count(),
            "embedding_model": self.config.rag_embedding_model,
            "embedding_dimension": self.embedding_generator.get_embedding_dimension(),
            "collection_name": self.config.rag_collection_name,
            "fl_server_address": self.config.fl_server_address,
            "fl_num_rounds": self.config.fl_num_rounds,
            "fl_min_clients": self.config.fl_min_clients,
        }
    
    def print_stats(self):
        """Print system statistics."""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("Federated RAG System Statistics")
        print("=" * 60)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Embedding Model: {stats['embedding_model']}")
        print(f"Embedding Dimension: {stats['embedding_dimension']}")
        print(f"Collection Name: {stats['collection_name']}")
        print(f"\nFL Server Address: {stats['fl_server_address']}")
        print(f"FL Rounds: {stats['fl_num_rounds']}")
        print(f"FL Min Clients: {stats['fl_min_clients']}")
        print("=" * 60 + "\n")
    
    def clear_knowledge_base(self):
        """Clear all documents from the knowledge base."""
        print("Clearing knowledge base...")
        self.vector_store.clear()
        print("Knowledge base cleared")
