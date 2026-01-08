"""Integration layer combining FL and RAG."""

from .config import Config, get_config
from .fl_rag_system import FederatedRAGSystem

__all__ = ['Config', 'get_config', 'FederatedRAGSystem']
