"""Federated Learning Layer using Flower Framework."""

from .model import SimpleNet
from .client import FlowerClient
from .server import start_server
from .strategy import get_strategy

__all__ = ['SimpleNet', 'FlowerClient', 'start_server', 'get_strategy']
