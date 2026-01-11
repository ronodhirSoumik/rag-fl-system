# Federated Learning with RAG

A comprehensive implementation combining **Federated Learning** using the Flower Framework with **Retrieval Augmented Generation (RAG)** for privacy-preserving, knowledge-enhanced AI systems.

## 🎯 Project Overview

This project demonstrates how to:
- Train machine learning models across distributed clients using Flower Framework
- Implement a RAG system for enhanced information retrieval
- Combine federated learning with RAG for privacy-preserving knowledge systems

## 🏗️ Architecture

```
federated-learning-cosys/
├── fl_layer/              # Federated Learning components
│   ├── server.py          # FL server implementation
│   ├── client.py          # FL client implementation
│   ├── strategy.py        # Training strategy
│   └── model.py           # Model definition
├── rag_layer/             # RAG components
│   ├── document_loader.py # Document processing
│   ├── embeddings.py      # Embedding generation
│   ├── vector_store.py    # Vector database
│   └── retriever.py       # Retrieval mechanism
├── integration/           # Integration layer
│   ├── fl_rag_system.py   # Combined system
│   └── config.py          # Configuration
├── examples/              # Usage examples
│   ├── run_fl_training.py
│   └── run_rag_query.py
├── data/                  # Sample data
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Federated Learning

```bash
# Start FL server
python examples/run_fl_training.py --mode server

# Start FL clients (in separate terminals)
python examples/run_fl_training.py --mode client --client-id 1
python examples/run_fl_training.py --mode client --client-id 2
```

### Using RAG System

```bash
# Run RAG query example
python examples/run_rag_query.py
```

## 📚 Features

### Federated Learning Layer
- ✅ Flower Framework integration
- ✅ Customizable training strategies
- ✅ Support for multiple clients
- ✅ Model aggregation and evaluation

### RAG Layer
- ✅ Document loading and chunking
- ✅ Sentence embeddings with transformers
- ✅ Vector store with ChromaDB/FAISS
- ✅ Semantic search and retrieval
- ✅ Context-aware query answering

## 🔧 Configuration

Edit `integration/config.py` to customize:
- Number of federated rounds
- Client selection strategy
- Embedding model
- Vector store settings
- Document chunk size

## 📖 Documentation

For detailed documentation, see:
- [Flower Framework Docs](https://flower.dev/docs/)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
