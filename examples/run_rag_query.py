"""Example script for RAG query demonstration."""

import sys
import argparse

# Add parent directory to path
sys.path.insert(0, '..')

from integration import FederatedRAGSystem, get_config, print_config


def run_interactive_mode(system: FederatedRAGSystem):
    """Run interactive query mode.
    
    Args:
        system: FederatedRAGSystem instance
    """
    print("\n" + "=" * 60)
    print("Interactive RAG Query Mode")
    print("=" * 60)
    print("Type your queries below. Type 'quit' or 'exit' to stop.")
    print("Type 'stats' to see system statistics.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive mode...")
                break
            
            if query.lower() == 'stats':
                system.print_stats()
                continue
            
            if not query:
                continue
            
            # Query the knowledge base
            context = system.query_knowledge_base(query)
            
            print("\n" + "-" * 60)
            print("Retrieved Context:")
            print("-" * 60)
            print(context)
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def run_demo_queries(system: FederatedRAGSystem):
    """Run demonstration queries.
    
    Args:
        system: FederatedRAGSystem instance
    """
    demo_queries = [
        "What is federated learning?",
        "How does privacy work in federated learning?",
        "What are the advantages of federated learning?",
        "Explain model aggregation",
        "What is differential privacy?"
    ]
    
    print("\n" + "=" * 60)
    print("Running Demo Queries")
    print("=" * 60 + "\n")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n[Query {i}/{len(demo_queries)}] {query}")
        print("-" * 60)
        
        results = system.get_retrieval_results(query, top_k=3)
        
        if results:
            for j, result in enumerate(results, 1):
                print(f"\n  Result {j}:")
                print(f"  Score: {result.score:.4f}")
                print(f"  Content: {result.content[:200]}...")
                if result.metadata:
                    print(f"  Source: {result.metadata.get('filename', 'Unknown')}")
        else:
            print("  No results found")
        
        print("-" * 60)


def main():
    """Main entry point for RAG query script."""
    parser = argparse.ArgumentParser(
        description="RAG Query Demonstration"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "demo"],
        default="demo",
        help="Mode to run: interactive or demo"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/sample_documents.txt",
        help="Path to documents to load"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing knowledge base before loading"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    print("\n" + "=" * 60)
    print("RAG Query System")
    print("=" * 60)
    
    print_config(config)
    
    # Initialize system
    print("\nInitializing Federated RAG System...")
    system = FederatedRAGSystem(config)
    
    # Clear if requested
    if args.clear:
        system.clear_knowledge_base()
    
    # Load knowledge base if empty or if clearing
    if system.retriever.get_document_count() == 0 or args.clear:
        import os
        if os.path.exists(args.data_path):
            print(f"\nLoading knowledge base from: {args.data_path}")
            system.load_knowledge_base(args.data_path)
        else:
            print(f"\nWarning: Data file not found: {args.data_path}")
            print("Creating sample knowledge base...")
            
            # Create sample data if file doesn't exist
            from rag_layer import Document
            sample_docs = [
                Document(
                    "Federated learning is a machine learning approach that trains models across decentralized devices or servers holding local data samples, without exchanging them. This approach ensures data privacy and security.",
                    {"source": "sample", "topic": "federated_learning"}
                ),
                Document(
                    "In federated learning, model aggregation combines the locally trained models from different clients into a global model. The most common aggregation method is Federated Averaging (FedAvg).",
                    {"source": "sample", "topic": "aggregation"}
                ),
                Document(
                    "Privacy in federated learning is maintained because raw data never leaves the local devices. Only model updates are shared with the central server, protecting sensitive information.",
                    {"source": "sample", "topic": "privacy"}
                ),
            ]
            system.retriever.add_documents(sample_docs)
            print("Sample knowledge base created")
    
    # Show statistics
    system.print_stats()
    
    # Run in selected mode
    if args.mode == "interactive":
        run_interactive_mode(system)
    elif args.mode == "demo":
        run_demo_queries(system)


if __name__ == "__main__":
    main()
