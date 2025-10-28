"""Neural network models available for the traffic project."""

from .trainer import (
    NeuralModelResult,
    NeuralTrainingBundle,
    TrafficNeuralTrainer,
)

__all__ = [
    "TrafficNeuralTrainer",
    "NeuralTrainingBundle",
    "NeuralModelResult",
]
