"""Pipeline orchestrating data cleaning, visualisation and RL training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from src.data.data_cleaner import DataCleaner
from src.data.data_loader import DataLoader
from src.models.reinforcement import TrafficRLTrainer
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.visualization import TrafficPlotter


class TrafficWorkflow:
    """Co-ordinate the three stages required by the project."""

    def __init__(
        self,
        config: Optional[Config] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or Config()
        self.logger = logger or setup_logger(__name__)

        self.data_loader = DataLoader(data_path=str(self.config.data_path))
        self.cleaner = DataCleaner(logger=self.logger)
        self.plotter = TrafficPlotter(output_path=self.config.output_path, logger=self.logger)
        self.trainer = TrafficRLTrainer(logger=self.logger)

    def run(
        self,
        input_file: str,
        metric_column: Optional[str] = None,
        plot_prefix: Optional[str] = None,
    ) -> Dict[str, object]:
        """Execute the three stages of the workflow and return artefacts."""

        raw_data = self.data_loader.load_csv(input_file)
        cleaned_data, report = self.cleaner.clean(raw_data, dataset=input_file)

        prefix = plot_prefix or Path(input_file).stem
        plot_artifacts = self.plotter.create_descriptive_plots(cleaned_data, prefix)

        training_result = self.trainer.train_from_dataframe(
            cleaned_data, metric_column=metric_column
        )

        self.logger.info("Flujo completo ejecutado correctamente")

        return {
            "cleaned_data": cleaned_data,
            "cleaning_report": report.as_dict(),
            "plots": [artifact.path for artifact in plot_artifacts],
            "training": {
                "episodes": training_result.episodes,
                "metric_column": training_result.metric_column,
                "episode_rewards": training_result.episode_rewards,
                "epsilon_history": training_result.epsilon_history,
                "policy": training_result.greedy_policy().tolist(),
            },
        }


__all__ = ["TrafficWorkflow"]
