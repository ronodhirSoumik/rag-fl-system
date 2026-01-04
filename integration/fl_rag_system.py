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
    
    
