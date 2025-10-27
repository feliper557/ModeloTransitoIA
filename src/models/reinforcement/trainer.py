"""Simple reinforcement learning trainer for traffic signal control."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


class TrafficSignalEnv:
    """Toy environment representing the decision of a traffic signal."""

    def __init__(self, demand_samples: np.ndarray, num_bins: int = 10, max_steps: int = 50):
        if demand_samples.size == 0:
            raise ValueError("Se requieren muestras de demanda para crear el entorno")

        self.demand_samples = demand_samples
        self.num_bins = int(num_bins)
        self.max_steps = int(max_steps)
        self.action_space = np.array([-1, 0, 1])
        self.current_step = 0
        self.state = 0

    @property
    def action_size(self) -> int:
        return self.action_space.size

    def reset(self) -> int:
        self.current_step = 0
        self.state = int(np.random.randint(0, self.num_bins))
        return self.state

    def _sample_demand(self) -> int:
        raw_sample = float(np.random.choice(self.demand_samples))
        demand_bin = int(np.clip(round(raw_sample * (self.num_bins - 1)), 0, self.num_bins - 1))
        return demand_bin

    def step(self, action_index: int) -> tuple[int, float, bool]:
        self.current_step += 1

        adjustment = int(self.action_space[action_index])
        self.state = int(np.clip(self.state + adjustment, 0, self.num_bins - 1))

        demand_bin = self._sample_demand()
        reward = -abs(self.state - demand_bin)

        done = self.current_step >= self.max_steps
        return self.state, float(reward), bool(done)


@dataclass
class TrainingResult:
    """Container with the artefacts obtained after training."""

    episodes: int
    q_table: np.ndarray
    episode_rewards: List[float]
    epsilon_history: List[float]
    metric_column: str

    def greedy_policy(self) -> np.ndarray:
        """Return the greedy policy extracted from the Q-table."""

        return np.argmax(self.q_table, axis=1)


class TrafficRLTrainer:
    """Train a Q-learning agent using aggregated traffic intensity samples."""

    def __init__(
        self,
        episodes: int = 300,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        num_bins: int = 10,
        max_steps: int = 50,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_bins = num_bins
        self.max_steps = max_steps
        self.logger = logger or logging.getLogger(__name__)

    def train_from_dataframe(
        self, data: pd.DataFrame, metric_column: Optional[str] = None
    ) -> TrainingResult:
        """Prepare the environment using the dataframe and train the agent."""

        if data.empty:
            raise ValueError("El DataFrame proporcionado está vacío")

        column = self._select_metric_column(data, metric_column)

        series = data[column].astype(float)
        normalised = self._normalise_series(series)

        environment = TrafficSignalEnv(
            demand_samples=normalised.values,
            num_bins=self.num_bins,
            max_steps=self.max_steps,
        )

        return self._train(environment, column)

    # ------------------------------------------------------------------
    def _train(self, environment: TrafficSignalEnv, metric_column: str) -> TrainingResult:
        q_table = np.zeros((environment.num_bins, environment.action_size))
        epsilon = self.epsilon

        rewards: List[float] = []
        epsilon_history: List[float] = []

        for episode in range(self.episodes):
            state = environment.reset()
            total_reward = 0.0

            for _ in range(environment.max_steps):
                if np.random.rand() < epsilon:
                    action = np.random.randint(environment.action_size)
                else:
                    action = int(np.argmax(q_table[state]))

                next_state, reward, done = environment.step(action)

                best_next_action = np.max(q_table[next_state])
                q_table[state, action] = (1 - self.alpha) * q_table[state, action] + self.alpha * (
                    reward + self.gamma * best_next_action
                )

                state = next_state
                total_reward += reward

                if done:
                    break

            rewards.append(total_reward)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            if (episode + 1) % 50 == 0 or episode == 0:
                self.logger.info(
                    "Episodio %d/%d - Recompensa acumulada: %.2f - Epsilon: %.3f",
                    episode + 1,
                    self.episodes,
                    total_reward,
                    epsilon,
                )

        return TrainingResult(
            episodes=self.episodes,
            q_table=q_table,
            episode_rewards=rewards,
            epsilon_history=epsilon_history,
            metric_column=metric_column,
        )

    def _select_metric_column(self, data: pd.DataFrame, metric_column: Optional[str]) -> str:
        if metric_column and metric_column in data.columns:
            return metric_column

        numeric_columns = data.select_dtypes(include=["number"]).columns
        if numeric_columns.empty:
            raise ValueError(
                "No se encontraron columnas numéricas para entrenar el modelo de refuerzo"
            )

        if metric_column and metric_column not in data.columns:
            raise ValueError(f"La columna '{metric_column}' no existe en los datos proporcionados")

        selected = numeric_columns[0]
        self.logger.warning(
            "Columna '%s' no encontrada. Se utilizará '%s' por defecto.",
            metric_column,
            selected,
        )
        return selected

    @staticmethod
    def _normalise_series(series: pd.Series) -> pd.Series:
        min_value = float(series.min())
        max_value = float(series.max())

        if max_value - min_value == 0:
            return pd.Series(np.zeros_like(series), index=series.index)

        return (series - min_value) / (max_value - min_value)


__all__ = ["TrafficRLTrainer", "TrainingResult"]
