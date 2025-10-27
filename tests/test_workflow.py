"""Integration tests for the full workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.pipeline import TrafficWorkflow


class _TestConfig:
    def __init__(self, data_path: Path, output_path: Path) -> None:
        self._data_path = data_path
        self._output_path = output_path

    @property
    def data_path(self) -> Path:  # type: ignore[override]
        return self._data_path

    @property
    def output_path(self) -> Path:  # type: ignore[override]
        return self._output_path


def test_workflow_runs_three_stages(tmp_path):
    data_path = tmp_path / "data"
    output_path = tmp_path / "output"
    data_path.mkdir()
    output_path.mkdir()

    df = pd.DataFrame(
        {
            "fecha": pd.date_range("2023-01-01", periods=10, freq="D"),
            "volumen": range(10),
        }
    )
    csv_path = data_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    config = _TestConfig(data_path=data_path, output_path=output_path)
    workflow = TrafficWorkflow(config=config)

    result = workflow.run("sample.csv", metric_column="volumen", plot_prefix="test")

    assert result["cleaning_report"]["final_rows"] == 10
    assert len(result["plots"]) >= 1
    assert result["training"]["episodes"] == workflow.trainer.episodes
