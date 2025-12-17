"""Flower server implementation for federated learning."""

import flwr as fl
from typing import Optional

from .strategy import get_strategy, CustomFedAvg


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 5,
    min_clients: int = 2,
    use_custom_strategy: bool = False
):
    """Start the Flower federated learning server.
    
    Args:
        server_address: Address to bind the server to
        num_rounds: Number of federated learning rounds
        min_clients: Minimum number of clients required
        use_custom_strategy: Whether to use custom strategy with logging
    """
    print("=" * 60)
    print("Starting Flower Federated Learning Server")
    print("=" * 60)
    print(f"Server address: {server_address}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Minimum clients: {min_clients}")
    print("=" * 60)
    
    # Create strategy
    if use_custom_strategy:
        strategy = CustomFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
        )
    else:
        strategy = get_strategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
        )
    
    # Configure server
    config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )
    
    print("\n" + "=" * 60)
    print("Federated Learning Complete!")
    print("=" * 60)


def start_simulation(
    num_clients: int = 2,
    num_rounds: int = 5,
    client_fn=None
):
    """Start a federated learning simulation.
    
    This is useful for testing without setting up multiple processes.
    
    Args:
        num_clients: Number of simulated clients
        num_rounds: Number of federated learning rounds
        client_fn: Function to create client instances
    """
    print("=" * 60)
    print("Starting Flower Federated Learning Simulation")
    print("=" * 60)
    print(f"Number of clients: {num_clients}")
    print(f"Number of rounds: {num_rounds}")
    print("=" * 60)
    
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )
    
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
