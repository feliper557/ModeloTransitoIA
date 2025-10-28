"""Neural network trainers for supervised traffic learning tasks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class NeuralModelResult:
    """Container with the artefacts from training a neural model."""

    name: str
    metrics: Dict[str, float]
    loss_curve: List[float]
    y_true: np.ndarray
    y_pred: np.ndarray

    def as_dict(self) -> Dict[str, object]:
        """Return a serialisable representation."""

        return {
            "name": self.name,
            "metrics": self.metrics,
            "loss_curve": list(self.loss_curve),
            "y_true": self.y_true.tolist(),
            "y_pred": self.y_pred.tolist(),
        }


@dataclass
class NeuralTrainingBundle:
    """Grouping of all neural model results."""

    target_column: str
    models: List[NeuralModelResult]

    def best_model(self) -> Optional[NeuralModelResult]:
        """Return the model with the lowest mean squared error."""

        if not self.models:
            return None
        return min(self.models, key=lambda result: result.metrics.get("mse", np.inf))

    def as_dict(self) -> Dict[str, object]:
        """Return a serialisable representation of the bundle."""

        return {
            "target_column": self.target_column,
            "models": [model.as_dict() for model in self.models],
            "best_model": self.best_model().as_dict() if self.best_model() else None,
        }


class TrafficNeuralTrainer:
    """Train multiple neural networks for traffic prediction tasks."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)

    def train_from_dataframe(
        self, data: pd.DataFrame, target_column: str
    ) -> NeuralTrainingBundle:
        """Prepare features and train two neural networks."""

        if target_column not in data.columns:
            raise ValueError(
                f"La columna objetivo '{target_column}' no está presente en el DataFrame"
            )

        features, target = self._prepare_supervised_arrays(data, target_column)

        if features.shape[0] < 4:
            raise ValueError("Se requieren al menos 4 muestras para entrenar las redes neuronales")

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        shallow_result = self._train_regressor(
            name="MLPRegressor_shallow",
            hidden_layers=(32,),
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        deep_result = self._train_regressor(
            name="MLPRegressor_deep",
            hidden_layers=(64, 32),
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        return NeuralTrainingBundle(target_column=target_column, models=[shallow_result, deep_result])

    # ------------------------------------------------------------------
    def _train_regressor(
        self,
        name: str,
        hidden_layers: Tuple[int, ...],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> NeuralModelResult:
        """Train an ``MLPRegressor`` and compute standard metrics."""

        enable_early_stopping = X_train.shape[0] >= 20

        regressor = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layers,
                        activation="relu",
                        learning_rate_init=1e-3,
                        max_iter=500,
                        random_state=self.random_state,
                        early_stopping=enable_early_stopping,
                        n_iter_no_change=20,
                        validation_fraction=0.2,
                    ),
                ),
            ]
        )

        regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_test)

        mse = float(mean_squared_error(y_test, predictions))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        mlp_stage = regressor.named_steps["mlp"]
        loss_curve = list(getattr(mlp_stage, "loss_curve_", []))

        self.logger.info(
            "Modelo %s entrenado. MSE: %.4f, MAE: %.4f, R2: %.4f",
            name,
            mse,
            mae,
            r2,
        )

        return NeuralModelResult(
            name=name,
            metrics={"mse": mse, "mae": mae, "r2": r2},
            loss_curve=loss_curve,
            y_true=y_test,
            y_pred=predictions,
        )

    def _prepare_supervised_arrays(
        self, data: pd.DataFrame, target_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate model-ready arrays from the dataframe."""

        df = data.copy()

        # Ensure datetime columns are expanded into meaningful numerical features
        for column in df.columns:
            if column == target_column:
                continue

            if pd.api.types.is_datetime64_any_dtype(df[column]):
                df = self._expand_datetime(df, column)
            elif pd.api.types.is_object_dtype(df[column]):
                parsed = pd.to_datetime(df[column], errors="coerce")
                if parsed.notna().sum() >= len(parsed) * 0.5:
                    df[column] = parsed
                    df = self._expand_datetime(df, column)

        # Remove the original datetime columns after expansion
        datetime_columns = [
            column
            for column in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[column]) and column != target_column
        ]
        df = df.drop(columns=datetime_columns)

        # Separate target
        y = df[target_column].astype(float).values
        X_df = df.drop(columns=[target_column])

        # Encode categorical columns
        categorical_columns = X_df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_columns) > 0:
            encoder = OneHotEncoder(drop="if_binary", sparse=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(X_df[categorical_columns])
            encoded_columns = encoder.get_feature_names_out(categorical_columns)
            encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=X_df.index)
            X_df = pd.concat([X_df.drop(columns=categorical_columns), encoded_df], axis=1)

        numeric_df = X_df.select_dtypes(include=["number"]).copy()

        if numeric_df.empty:
            # As a fallback, use the index as a sequential feature
            numeric_df["sequential_index"] = np.arange(len(X_df))

        numeric_df = numeric_df.ffill().bfill().fillna(0.0)

        features = numeric_df.to_numpy(dtype=float)
        return features, y

    def _expand_datetime(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Expand a datetime column into cyclical and calendar features."""

        dt_series = pd.to_datetime(df[column], errors="coerce")
        base_name = column

        df[f"{base_name}_year"] = dt_series.dt.year.fillna(0).astype(int)
        df[f"{base_name}_month"] = dt_series.dt.month.fillna(0).astype(int)
        df[f"{base_name}_day"] = dt_series.dt.day.fillna(0).astype(int)
        df[f"{base_name}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)

        # Cyclical encoding for month and day of week to respect periodicity
        df[f"{base_name}_month_sin"] = np.sin(2 * np.pi * df[f"{base_name}_month"] / 12)
        df[f"{base_name}_month_cos"] = np.cos(2 * np.pi * df[f"{base_name}_month"] / 12)
        df[f"{base_name}_dow_sin"] = np.sin(2 * np.pi * df[f"{base_name}_dayofweek"] / 7)
        df[f"{base_name}_dow_cos"] = np.cos(2 * np.pi * df[f"{base_name}_dayofweek"] / 7)

        return df


__all__ = [
    "TrafficNeuralTrainer",
    "NeuralModelResult",
    "NeuralTrainingBundle",
]
