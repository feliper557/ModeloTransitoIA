"""Simple reinforcement learning trainer for traffic signal control."""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


class SimpleQNetwork:
    """Small fully connected network trained with vanilla gradient descent."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: int = 32,
        learning_rate: float = 1e-3,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

        limit1 = np.sqrt(6.0 / (input_dim + hidden_units))
        limit2 = np.sqrt(6.0 / (hidden_units + output_dim))

        rng = np.random.default_rng()
        self.w1 = rng.uniform(-limit1, limit1, size=(input_dim, hidden_units)).astype(np.float32)
        self.b1 = np.zeros(hidden_units, dtype=np.float32)
        self.w2 = rng.uniform(-limit2, limit2, size=(hidden_units, output_dim)).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def predict(self, states: np.ndarray) -> np.ndarray:
        _, _, output = self._forward(states)
        return output

    def train_on_batch(self, states: np.ndarray, targets: np.ndarray) -> None:
        pre_activations, activations, predictions = self._forward(states)

        batch_size = states.shape[0]
        error = predictions - targets
        grad_output = (2.0 / batch_size) * error

        grad_w2 = activations.T @ grad_output
        grad_b2 = grad_output.sum(axis=0)

        grad_hidden = grad_output @ self.w2.T
        relu_mask = (pre_activations > 0).astype(np.float32)
        grad_hidden *= relu_mask

        grad_w1 = states.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)

        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1

    def copy_weights_from(self, other: "SimpleQNetwork") -> None:
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()

    def clone(self) -> "SimpleQNetwork":
        clone = SimpleQNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_units=self.hidden_units,
            learning_rate=self.learning_rate,
        )
        clone.copy_weights_from(self)
        return clone

    def _forward(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pre_activation = states @ self.w1 + self.b1
        activation = np.maximum(pre_activation, 0.0)
        output = activation @ self.w2 + self.b2
        return pre_activation, activation, output


class TrafficRLTrainer:
    """Train a DQN agent using aggregated traffic intensity samples."""

    def __init__(
        self,
        episodes: int = 120,
        gamma: float = 0.95,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        num_bins: int = 10,
        max_steps: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        replay_buffer_size: int = 5000,
        min_replay_size: int = 64,
        target_update_frequency: int = 10,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_bins = num_bins
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.replay_buffer_size = replay_buffer_size
        self.min_replay_size = min_replay_size
        self.target_update_frequency = target_update_frequency
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
        main_network = self._build_network(environment.action_size)
        target_network = main_network.clone()

        epsilon = self.epsilon
        rewards: List[float] = []
        epsilon_history: List[float] = []
        replay_buffer: deque[Tuple[int, int, float, int, bool]] = deque(maxlen=self.replay_buffer_size)

        for episode in range(self.episodes):
            state = environment.reset()
            total_reward = 0.0

            for _ in range(environment.max_steps):
                if np.random.rand() < epsilon:
                    action = int(np.random.randint(environment.action_size))
                else:
                    state_input = self._encode_state(state)
                    q_values = main_network.predict(state_input)[0]
                    action = int(np.argmax(q_values))

                next_state, reward, done = environment.step(action)
                total_reward += reward

                replay_buffer.append((state, action, reward, next_state, done))
                state = next_state

                if len(replay_buffer) >= self.min_replay_size:
                    batch_size = min(self.batch_size, len(replay_buffer))
                    batch = random.sample(replay_buffer, batch_size)
                    self._train_batch(batch, main_network, target_network)

                if done:
                    break

            if (episode + 1) % self.target_update_frequency == 0:
                target_network.copy_weights_from(main_network)

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

        state_identity = np.eye(environment.num_bins, dtype=np.float32)
        q_table = main_network.predict(state_identity)

        return TrainingResult(
            episodes=self.episodes,
            q_table=q_table,
            episode_rewards=rewards,
            epsilon_history=epsilon_history,
            metric_column=metric_column,
        )

    def _build_network(self, action_size: int) -> SimpleQNetwork:
        return SimpleQNetwork(
            input_dim=self.num_bins,
            output_dim=action_size,
            hidden_units=32,
            learning_rate=self.learning_rate,
        )

    def _encode_state(self, state: int) -> np.ndarray:
        one_hot = np.zeros((1, self.num_bins), dtype=np.float32)
        one_hot[0, state] = 1.0
        return one_hot

    def _train_batch(
        self,
        batch: List[Tuple[int, int, float, int, bool]],
        main_network: SimpleQNetwork,
        target_network: SimpleQNetwork,
    ) -> None:
        states = np.vstack([self._encode_state(transition[0]) for transition in batch])
        next_states = np.vstack([self._encode_state(transition[3]) for transition in batch])
        actions = np.array([transition[1] for transition in batch], dtype=np.int64)
        rewards = np.array([transition[2] for transition in batch], dtype=np.float32)
        dones = np.array([transition[4] for transition in batch], dtype=bool)

        current_qs = main_network.predict(states)
        future_qs = target_network.predict(next_states)

        targets = current_qs.copy()
        max_future_qs = np.max(future_qs, axis=1)
        targets[np.arange(len(batch)), actions] = rewards + (
            (1 - dones.astype(np.float32)) * self.gamma * max_future_qs
        )

        main_network.train_on_batch(states, targets)

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
