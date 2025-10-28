"""Utilities to create exploratory plots for the project."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Protocol

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class _HasMetrics(Protocol):
    name: str
    metrics: dict
    loss_curve: Sequence[float]
    y_true: Sequence[float]
    y_pred: Sequence[float]


class _RLResult(Protocol):
    episode_rewards: Sequence[float]

@dataclass
class PlotArtifact:
    """Information about a generated plot."""

    description: str
    path: Path


class TrafficPlotter:
    """Create a set of descriptive plots for traffic datasets."""

    def __init__(self, output_path: Path, logger: logging.Logger | None = None):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)

    def create_descriptive_plots(
        self, data: pd.DataFrame, prefix: str = "traffic"
    ) -> List[PlotArtifact]:
        """Generate a standard collection of descriptive plots."""

        if data.empty:
            raise ValueError("No se pueden generar gráficas para un DataFrame vacío")

        artifacts: List[PlotArtifact] = []
        numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()

        self.logger.debug("Columnas numéricas detectadas: %s", numeric_columns)

        if numeric_columns:
            artifacts.extend(
                self._plot_distributions(data[numeric_columns], prefix=prefix)
            )
            if len(numeric_columns) > 1:
                artifacts.append(self._plot_correlation(data[numeric_columns], prefix))
        else:
            self.logger.warning(
                "No se encontraron columnas numéricas para graficar distribuciones"
            )

        if "fecha" in data.columns:
            artifacts.append(self._plot_time_series(data, prefix=prefix))

        return artifacts

    def create_model_plots(
        self,
        neural_models: Sequence[_HasMetrics],
        rl_result: Optional[_RLResult],
        prefix: str = "traffic",
    ) -> List[PlotArtifact]:
        """Generate plots comparing the three learning approaches."""

        artifacts: List[PlotArtifact] = []

        if neural_models:
            artifacts.append(self._plot_neural_losses(neural_models, prefix))
            for model in neural_models:
                artifacts.append(self._plot_neural_predictions(model, prefix))

        if rl_result:
            artifacts.append(self._plot_reinforcement_rewards(rl_result, prefix))

        if neural_models and rl_result:
            artifacts.append(self._plot_model_scoreboard(neural_models, rl_result, prefix))

        return artifacts

    # ------------------------------------------------------------------
    def _plot_distributions(self, data: pd.DataFrame, prefix: str) -> List[PlotArtifact]:
        artifacts: List[PlotArtifact] = []

        for column in data.columns:
            figure_path = self.output_path / f"{prefix}_{column}_hist.png"
            plt.figure(figsize=(8, 5))
            sns.histplot(data[column], kde=True, color="#0077b6")
            plt.title(f"Distribución de {column}")
            plt.xlabel(column)
            plt.ylabel("Frecuencia")
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()

            artifacts.append(
                PlotArtifact(
                    description=f"Histograma para {column}",
                    path=figure_path,
                )
            )
            self.logger.debug("Histograma generado: %s", figure_path)

        return artifacts

    def _plot_correlation(self, data: pd.DataFrame, prefix: str) -> PlotArtifact:
        figure_path = self.output_path / f"{prefix}_correlacion.png"
        plt.figure(figsize=(10, 8))
        correlation = data.corr(numeric_only=True)
        sns.heatmap(correlation, annot=True, cmap="viridis")
        plt.title("Matriz de correlación")
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()

        self.logger.debug("Matriz de correlación generada: %s", figure_path)

        return PlotArtifact(
            description="Matriz de correlación",
            path=figure_path,
        )

    def _plot_time_series(self, data: pd.DataFrame, prefix: str) -> PlotArtifact:
        figure_path = self.output_path / f"{prefix}_serie_tiempo.png"

        df = data.copy()
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"])

        numeric_columns = df.select_dtypes(include=["number"]).columns
        if numeric_columns.empty:
            raise ValueError(
                "Se requiere al menos una columna numérica para la serie de tiempo"
            )

        column = numeric_columns[0]

        df = df.sort_values("fecha")
        plt.figure(figsize=(10, 5))
        plt.plot(df["fecha"], df[column], marker="o", linestyle="-")
        plt.title(f"Serie de tiempo para {column}")
        plt.xlabel("Fecha")
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()

        self.logger.debug("Serie de tiempo generada: %s", figure_path)

        return PlotArtifact(
            description=f"Serie de tiempo para {column}",
            path=figure_path,
        )

    # ------------------------------------------------------------------
    def _plot_neural_losses(
        self, neural_models: Sequence[_HasMetrics], prefix: str
    ) -> PlotArtifact:
        figure_path = self.output_path / f"{prefix}_neural_losses.png"

        plt.figure(figsize=(10, 5))
        for model in neural_models:
            if model.loss_curve:
                plt.plot(model.loss_curve, label=model.name)

        if not plt.gca().has_data():
            plt.plot([], [])

        plt.title("Curvas de pérdida de las redes neuronales")
        plt.xlabel("Iteraciones")
        plt.ylabel("Pérdida")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()

        return PlotArtifact(
            description="Curvas de pérdida de redes neuronales",
            path=figure_path,
        )

    def _plot_neural_predictions(
        self, model: _HasMetrics, prefix: str
    ) -> PlotArtifact:
        figure_path = self.output_path / f"{prefix}_{model.name}_predicciones.png"

        y_true = list(model.y_true)
        y_pred = list(model.y_pred)
        x_axis = list(range(len(y_true)))

        plt.figure(figsize=(10, 5))
        plt.plot(x_axis, y_true, marker="o", label="Valor real")
        plt.plot(x_axis, y_pred, marker="x", label="Predicción")
        plt.title(f"Predicciones vs valores reales - {model.name}")
        plt.xlabel("Muestras de prueba")
        plt.ylabel("Valor")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()

        return PlotArtifact(
            description=f"Predicciones del modelo {model.name}",
            path=figure_path,
        )

    def _plot_reinforcement_rewards(
        self, result: _RLResult, prefix: str
    ) -> PlotArtifact:
        figure_path = self.output_path / f"{prefix}_rl_rewards.png"

        rewards = list(result.episode_rewards)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(rewards) + 1), rewards, marker="o", color="#8338ec")
        plt.title("Recompensa acumulada por episodio (RL)")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()

        return PlotArtifact(
            description="Recompensas del modelo de refuerzo",
            path=figure_path,
        )

    def _plot_model_scoreboard(
        self, neural_models: Sequence[_HasMetrics], rl_result: _RLResult, prefix: str
    ) -> PlotArtifact:
        figure_path = self.output_path / f"{prefix}_comparacion_modelos.png"

        model_names = [model.name for model in neural_models]
        mse_values = [model.metrics.get("mse", 0.0) for model in neural_models]
        average_reward = float(sum(rl_result.episode_rewards) / len(rl_result.episode_rewards))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(model_names, mse_values, color="#219ebc")
        axes[0].set_title("Error cuadrático medio (menor es mejor)")
        axes[0].set_ylabel("MSE")
        axes[0].tick_params(axis="x", rotation=20)

        axes[1].bar(["RL"], [average_reward], color="#ffb703")
        axes[1].set_title("Recompensa media del modelo de refuerzo")
        axes[1].set_ylabel("Recompensa media")

        plt.suptitle("Comparación de los tres enfoques de modelado")
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close(fig)

        return PlotArtifact(
            description="Comparación de métricas entre modelos",
            path=figure_path,
        )


__all__ = ["TrafficPlotter", "PlotArtifact"]
