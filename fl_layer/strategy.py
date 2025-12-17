"""Federated learning strategies for Flower."""

import flwr as fl
from typing import Optional, Dict, List, Tuple
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
        
    Returns:
        Aggregated metrics dictionary
    """
    # Calculate weighted averages
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0}


def get_strategy(
    fraction_fit: float = 0.5,
    fraction_evaluate: float = 0.5,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2
) -> fl.server.strategy.Strategy:
    """Create a federated learning strategy.
    
    Args:
        fraction_fit: Fraction of clients to use for training
        fraction_evaluate: Fraction of clients to use for evaluation
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum number of available clients
        
    Returns:
        Flower strategy instance
    """
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    return strategy


class CustomFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy with additional logging and configuration.
    
    This strategy extends the standard FedAvg with custom behavior
    for model aggregation and evaluation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_number = 0
        
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures
    ):
        """Aggregate training results with custom logging."""
        self.round_number = server_round
        print(f"\n[Round {server_round}] Aggregating training results...")
        print(f"  - Successful updates: {len(results)}")
        print(f"  - Failed updates: {len(failures)}")
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_metrics:
            print(f"  - Aggregated loss: {aggregated_metrics.get('loss', 'N/A')}")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures
    ):
        """Aggregate evaluation results with custom logging."""
        print(f"\n[Round {server_round}] Aggregating evaluation results...")
        print(f"  - Successful evaluations: {len(results)}")
        print(f"  - Failed evaluations: {len(failures)}")
        
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        if aggregated_metrics:
            print(f"  - Aggregated accuracy: {aggregated_metrics.get('accuracy', 'N/A'):.4f}")
        
        return aggregated_loss, aggregated_metrics
