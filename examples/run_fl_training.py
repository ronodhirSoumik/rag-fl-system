"""Example script for running federated learning training."""

import argparse
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add parent directory to path
sys.path.insert(0, '..')

from fl_layer import start_server, create_client
from integration import get_config, print_config


def create_dummy_data(num_samples=1000, client_id=0):
    """Create dummy MNIST-like data for demonstration.
    
    Args:
        num_samples: Number of samples to generate
        client_id: Client ID for data partitioning
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create synthetic data (28x28 images, 10 classes)
    np.random.seed(42 + client_id)
    
    # Training data
    train_images = torch.randn(num_samples, 1, 28, 28)
    train_labels = torch.randint(0, 10, (num_samples,))
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Test data
    test_images = torch.randn(num_samples // 5, 1, 28, 28)
    test_labels = torch.randint(0, 10, (num_samples // 5,))
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


def run_server(config):
    """Run the federated learning server.
    
    Args:
        config: Configuration object
    """
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING SERVER MODE")
    print("=" * 60)
    
    print_config(config)
    
    print("\nStarting server...")
    print("Waiting for clients to connect...")
    print(f"Server will run for {config.fl_num_rounds} rounds")
    print(f"Minimum {config.fl_min_clients} clients required")
    print("\nPress Ctrl+C to stop the server\n")
    
    start_server(
        server_address=config.fl_server_address,
        num_rounds=config.fl_num_rounds,
        min_clients=config.fl_min_clients,
        use_custom_strategy=True
    )


def run_client(client_id, config):
    """Run a federated learning client.
    
    Args:
        client_id: Unique client identifier
        config: Configuration object
    """
    import flwr as fl
    
    print("\n" + "=" * 60)
    print(f"FEDERATED LEARNING CLIENT MODE (Client {client_id})")
    print("=" * 60)
    
    print(f"\nClient ID: {client_id}")
    print(f"Server Address: {config.fl_server_address}")
    
    # Create dummy data for this client
    print(f"\nGenerating training data for client {client_id}...")
    train_loader, test_loader = create_dummy_data(
        num_samples=1000,
        client_id=client_id
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create client
    print("\nCreating Flower client...")
    client = create_client(client_id, train_loader, test_loader)
    
    # Connect to server
    print(f"Connecting to server at {config.fl_server_address}...")
    print("Press Ctrl+C to disconnect\n")
    
    fl.client.start_client(
        server_address=config.fl_server_address,
        client=client
    )
    
    print(f"\nClient {client_id} disconnected")


def run_simulation(config):
    """Run a federated learning simulation with multiple clients.
    
    Args:
        config: Configuration object
    """
    import flwr as fl
    
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING SIMULATION MODE")
    print("=" * 60)
    
    print_config(config)
    
    def client_fn(cid: str):
        """Create a client for simulation."""
        client_id = int(cid)
        train_loader, test_loader = create_dummy_data(
            num_samples=500,
            client_id=client_id
        )
        return create_client(client_id, train_loader, test_loader)
    
    print("\nStarting simulation...")
    print(f"Number of clients: {config.fl_min_clients}")
    print(f"Number of rounds: {config.fl_num_rounds}\n")
    
    from fl_layer import start_simulation
    start_simulation(
        num_clients=config.fl_min_clients,
        num_rounds=config.fl_num_rounds,
        client_fn=client_fn
    )


def main():
    """Main entry point for the FL training script."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Training with Flower Framework"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["server", "client", "simulation"],
        default="simulation",
        help="Mode to run: server, client, or simulation"
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=1,
        help="Client ID (only used in client mode)"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=None,
        help="Minimum number of clients"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Override config with command-line arguments
    if args.num_rounds is not None:
        config.fl_num_rounds = args.num_rounds
    if args.min_clients is not None:
        config.fl_min_clients = args.min_clients
    
    # Run in selected mode
    if args.mode == "server":
        run_server(config)
    elif args.mode == "client":
        run_client(args.client_id, config)
    elif args.mode == "simulation":
        run_simulation(config)


if __name__ == "__main__":
    main()
