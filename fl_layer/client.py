"""Flower client implementation for federated learning."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from typing import Dict, List, Tuple

from .model import SimpleNet, get_model_parameters, set_model_parameters


class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning.
    
    This client handles local training, evaluation, and parameter updates
    for federated learning using the Flower framework.
    """
    
    def __init__(
        self,
        client_id: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        device: str = "cpu"
    ):
        """Initialize the Flower client.
        
        Args:
            client_id: Unique identifier for this client
            trainloader: DataLoader for training data
            testloader: DataLoader for test data
            device: Device to use for training ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.model = SimpleNet().to(device)
        
    def get_parameters(self, config: Dict) -> List:
        """Return current model parameters."""
        return get_model_parameters(self.model)
    
    def set_parameters(self, parameters: List) -> None:
        """Update model parameters."""
        set_model_parameters(self.model, parameters)
    
    def fit(
        self,
        parameters: List,
        config: Dict
    ) -> Tuple[List, int, Dict]:
        """Train the model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Training configuration
            
        Returns:
            Tuple of (updated parameters, number of examples, metrics)
        """
        self.set_parameters(parameters)
        
        # Training configuration
        epochs = config.get("epochs", 1)
        learning_rate = config.get("learning_rate", 0.001)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Train the model
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return (
            get_model_parameters(self.model),
            len(self.trainloader.dataset),
            {"loss": avg_loss}
        )
    
    def evaluate(
        self,
        parameters: List,
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate the model on local test data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, number of examples, metrics)
        """
        self.set_parameters(parameters)
        
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.testloader) if len(self.testloader) > 0 else 0.0
        
        return avg_loss, total, {"accuracy": accuracy}


def create_client(client_id: int, trainloader: DataLoader, testloader: DataLoader):
    """Factory function to create a Flower client.
    
    Args:
        client_id: Unique identifier for the client
        trainloader: Training data loader
        testloader: Test data loader
        
    Returns:
        FlowerClient instance
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return FlowerClient(client_id, trainloader, testloader, device)
