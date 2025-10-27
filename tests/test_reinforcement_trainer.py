"""Tests for the reinforcement learning trainer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.reinforcement import TrafficRLTrainer


def test_trainer_produces_q_table():
    trainer = TrafficRLTrainer(episodes=5, max_steps=5, num_bins=5)
    df = pd.DataFrame({"flujo": np.linspace(0, 100, num=20)})

    result = trainer.train_from_dataframe(df, metric_column="flujo")

    assert result.q_table.shape[0] == 5
    assert len(result.episode_rewards) == 5
