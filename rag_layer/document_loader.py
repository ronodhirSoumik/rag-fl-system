"""Document loading and chunking for RAG."""

import os
from typing import List, Dict, Optional
from pathlib import Path
import re


class Document:
    """Represents a document chunk with metadata."""
    
    def __init__(self, content: str, metadata: Optional[Dict] = None):
        """Initialize a document.
        
        Args:
            content: Text content of the document
            metadata: Optional metadata dictionary
        """
        self.content = content
        self.metadata = metadata or {}
        
    def __repr__(self):
        return f"Document(content='{self.content[:50]}...', metadata={self.metadata})"


