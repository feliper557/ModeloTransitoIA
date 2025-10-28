"""Tests for the neural network trainer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.neural import TrafficNeuralTrainer


def test_neural_trainer_returns_two_models():
    trainer = TrafficNeuralTrainer(test_size=0.3, random_state=0)

    base_date = pd.Timestamp("2023-01-01")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "fecha": [base_date + pd.Timedelta(days=i) for i in range(30)],
            "volumen": np.linspace(10, 40, num=30) + rng.normal(0, 1, size=30),
            "temperatura": np.linspace(15, 20, num=30),
        }
    )

    result = trainer.train_from_dataframe(df, target_column="volumen")

    assert result.target_column == "volumen"
    assert len(result.models) == 2
    for model in result.models:
        assert "mse" in model.metrics
        assert len(model.y_true) == len(model.y_pred)

    assert result.best_model() is not None
