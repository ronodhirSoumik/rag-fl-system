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


class DocumentLoader:
    """Load and process documents from various sources."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the document loader.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def load_text_file(self, file_path: str) -> str:
        """Load content from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File content as string
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Document objects
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence ending punctuation
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end)
                )
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata['chunk_index'] = len(chunks)
                chunk_metadata['start_char'] = start
                chunk_metadata['end_char'] = end
                
                chunks.append(Document(chunk_text, chunk_metadata))
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def load_and_chunk(self, file_path: str) -> List[Document]:
        """Load a file and split it into chunks.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document chunks
        """
        content = self.load_text_file(file_path)
        metadata = {
            'source': file_path,
            'filename': os.path.basename(file_path)
        }
        return self.chunk_text(content, metadata)
    
    def load_directory(self, directory_path: str, pattern: str = "*.txt") -> List[Document]:
        """Load all files matching pattern from a directory.
        
        Args:
            directory_path: Path to the directory
            pattern: File pattern to match (default: *.txt)
            
        Returns:
            List of Document chunks from all files
        """
        all_documents = []
        directory = Path(directory_path)
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                documents = self.load_and_chunk(str(file_path))
                all_documents.extend(documents)
        
        return all_documents


def load_documents(
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    is_directory: bool = False
) -> List[Document]:
    """Convenience function to load documents.
    
    Args:
        source: File path or directory path
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        is_directory: Whether source is a directory
        
    Returns:
        List of Document chunks
    """
    loader = DocumentLoader(chunk_size, chunk_overlap)
    
    if is_directory:
        return loader.load_directory(source)
    else:
        return loader.load_and_chunk(source)
