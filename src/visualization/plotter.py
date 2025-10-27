"""Utilities to create exploratory plots for the project."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


__all__ = ["TrafficPlotter", "PlotArtifact"]
