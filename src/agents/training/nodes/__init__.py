"""Training Node Infrastructure.

Client/server architecture for distributed training data generation.
Allows generation to be offloaded to remote machines with GPUs.
"""

from agents.training.nodes.node_client import TrainingNodeClient
from agents.training.nodes.node_server import TrainingNodeServer

__all__ = [
    "TrainingNodeClient",
    "TrainingNodeServer",
]
