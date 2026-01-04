"""Configuration management for FL and RAG systems."""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for Federated Learning and RAG system."""
    
    # Federated Learning Configuration
    fl_server_address: str = "0.0.0.0:8080"
    fl_num_rounds: int = 5
    fl_min_clients: int = 2
    fl_learning_rate: float = 0.001
    fl_local_epochs: int = 1
    fl_batch_size: int = 32
    
    # RAG Configuration
    rag_embedding_model: str = "all-MiniLM-L6-v2"
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50
    rag_top_k: int = 5
    rag_collection_name: str = "federated_learning_docs"
    
    # Storage Configuration
    vector_store_path: Optional[str] = None
    data_directory: str = "./data"
    
    # Model Configuration
    model_input_channels: int = 1
    model_num_classes: int = 10
    
    def __post_init__(self):
        """Initialize paths after dataclass initialization."""
        if self.vector_store_path is None:
            self.vector_store_path = os.path.join(
                self.data_directory,
                "vector_store"
            )
        
        # Create necessary directories
        os.makedirs(self.data_directory, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)


def get_config() -> Config:
    """Get configuration with environment variable overrides.
    
    Returns:
        Config instance with settings
    """
    config = Config()
    
    # Override with environment variables if present
    if os.getenv("FL_SERVER_ADDRESS"):
        config.fl_server_address = os.getenv("FL_SERVER_ADDRESS")
    
    if os.getenv("FL_NUM_ROUNDS"):
        config.fl_num_rounds = int(os.getenv("FL_NUM_ROUNDS"))
    
    if os.getenv("FL_MIN_CLIENTS"):
        config.fl_min_clients = int(os.getenv("FL_MIN_CLIENTS"))
    
    if os.getenv("RAG_EMBEDDING_MODEL"):
        config.rag_embedding_model = os.getenv("RAG_EMBEDDING_MODEL")
    
    if os.getenv("RAG_TOP_K"):
        config.rag_top_k = int(os.getenv("RAG_TOP_K"))
    
    if os.getenv("DATA_DIRECTORY"):
        config.data_directory = os.getenv("DATA_DIRECTORY")
    
    return config


def print_config(config: Config):
    """Print configuration in a readable format.
    
    Args:
        config: Config instance to print
    """
    print("\n" + "=" * 60)
    print("Configuration Settings")
    print("=" * 60)
    
    print("\nFederated Learning:")
    print(f"  Server Address: {config.fl_server_address}")
    print(f"  Number of Rounds: {config.fl_num_rounds}")
    print(f"  Minimum Clients: {config.fl_min_clients}")
    print(f"  Learning Rate: {config.fl_learning_rate}")
    print(f"  Local Epochs: {config.fl_local_epochs}")
    print(f"  Batch Size: {config.fl_batch_size}")
    
    print("\nRAG System:")
    print(f"  Embedding Model: {config.rag_embedding_model}")
    print(f"  Chunk Size: {config.rag_chunk_size}")
    print(f"  Chunk Overlap: {config.rag_chunk_overlap}")
    print(f"  Top-K Results: {config.rag_top_k}")
    print(f"  Collection Name: {config.rag_collection_name}")
    
    print("\nStorage:")
    print(f"  Vector Store Path: {config.vector_store_path}")
    print(f"  Data Directory: {config.data_directory}")
    
    print("=" * 60 + "\n")
