"""Retriever for semantic search and document retrieval."""

from typing import List, Dict, Optional, Tuple
import numpy as np

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .document_loader import Document


class RetrievalResult:
    """Represents a retrieval result with document and score."""
    
    def __init__(self, content: str, metadata: Dict, score: float):
        """Initialize a retrieval result.
        
        Args:
            content: Document content
            metadata: Document metadata
            score: Similarity score (lower is better for distance metrics)
        """
        self.content = content
        self.metadata = metadata
        self.score = score
        
    def __repr__(self):
        return f"RetrievalResult(score={self.score:.4f}, content='{self.content[:50]}...')"


class Retriever:
    """Retriever for semantic search over documents."""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore
    ):
        """Initialize the retriever.
        
        Args:
            embedding_generator: Embedding generator instance
            vector_store: Vector store instance
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the retriever.
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            print("No documents to add")
            return
        
        print(f"Generating embeddings for {len(documents)} documents...")
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_generator.embed_batch(texts)
        
        print("Adding documents to vector store...")
        self.vector_store.add_documents(documents, embeddings)
        print("Documents added successfully")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of RetrievalResult objects
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)
        
        # Query vector store
        documents, metadatas, distances = self.vector_store.query(
            query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        # Create retrieval results
        results = []
        for doc, metadata, distance in zip(documents, metadatas, distances):
            # Convert distance to similarity score (1 - normalized distance)
            # ChromaDB returns L2 distance, so lower is better
            results.append(RetrievalResult(doc, metadata, distance))
        
        return results
    
    def get_context(
        self,
        query: str,
        top_k: int = 3,
        separator: str = "\n\n"
    ) -> str:
        """Retrieve and format context for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            separator: Separator between documents
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=top_k)
        
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Context {i}] {result.content}")
        
        return separator.join(context_parts)
    
    def get_document_count(self) -> int:
        """Get the number of documents in the retriever.
        
        Returns:
            Number of documents
        """
        return self.vector_store.get_count()
