"""Data cleaning utilities for the traffic project."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_column_name(name: str) -> str:
    """Return a normalized column name in snake_case."""

    normalized = name.strip().lower().replace(" ", "_")
    normalized = normalized.replace("-", "_").replace("/", "_")
    return "".join(ch for ch in normalized if ch.isalnum() or ch == "_")


@dataclass
class CleaningReport:
    """Summary of the operations performed during data cleaning."""

    initial_rows: int
    final_rows: int
    columns_renamed: Dict[str, str]
    numeric_filled: Dict[str, float]
    categorical_filled: Dict[str, str]
    duplicates_removed: int
    remaining_missing: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        """Return the report as a serialisable dictionary."""

        return {
            "initial_rows": self.initial_rows,
            "final_rows": self.final_rows,
            "rows_removed": self.initial_rows - self.final_rows,
            "columns_renamed": self.columns_renamed,
            "numeric_filled": self.numeric_filled,
            "categorical_filled": self.categorical_filled,
            "duplicates_removed": self.duplicates_removed,
            "remaining_missing": self.remaining_missing,
        }


class DataCleaner:
    """Clean and prepare raw traffic datasets."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def clean(
        self, data: pd.DataFrame, dataset: Optional[str] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Clean the provided dataframe.

        Parameters
        ----------
        data:
            Raw dataframe loaded from disk.

        Returns
        -------
        Tuple[pd.DataFrame, CleaningReport]
            The cleaned dataframe and a report with the performed operations.
        """

        if data is None:
            raise ValueError("Se requiere un DataFrame para el proceso de limpieza")

        self.logger.debug("Iniciando limpieza de datos con %d filas", len(data))
        cleaned = data.copy()

        dataset_key = dataset.lower() if dataset else ""

        initial_rows = len(cleaned)

        columns_renamed = self._rename_columns(cleaned)

        cleaned = self._apply_dataset_specific_rules(cleaned, dataset_key)

        duplicate_subset = self._duplicate_subset(cleaned, dataset_key)
        duplicates_removed = self._remove_duplicates(cleaned, subset=duplicate_subset)

        numeric_filled, categorical_filled = self._fill_missing_values(cleaned)
        # Re-evaluate duplicates after imputing values that could make rows identical
        duplicates_removed += self._remove_duplicates(cleaned, subset=duplicate_subset)

        remaining_missing = cleaned.isna().sum().to_dict()
        final_rows = len(cleaned)

        report = CleaningReport(
            initial_rows=initial_rows,
            final_rows=final_rows,
            columns_renamed=columns_renamed,
            numeric_filled=numeric_filled,
            categorical_filled=categorical_filled,
            duplicates_removed=duplicates_removed,
            remaining_missing=remaining_missing,
        )

        self.logger.info(
            "Limpieza completada. Filas iniciales: %d, filas finales: %d",
            initial_rows,
            final_rows,
        )

        return cleaned, report

    def _rename_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Normalise column names to snake_case."""

        mapping: Dict[str, str] = {}
        new_columns: List[str] = []

        for column in df.columns:
            normalized = _normalize_column_name(column)
            mapping[column] = normalized
            new_columns.append(normalized)

        if mapping:
            self.logger.debug("Columnas renombradas: %s", mapping)
            df.columns = new_columns

        return mapping

    def _remove_duplicates(
        self, df: pd.DataFrame, subset: Optional[Iterable[str]] = None
    ) -> int:
        """Remove duplicate rows from the dataframe."""

        subset_list = list(subset) if subset else None
        duplicates = df.duplicated(subset=subset_list).sum()
        if duplicates:
            self.logger.debug("Eliminando %d filas duplicadas", duplicates)
            df.drop_duplicates(subset=subset_list, inplace=True)
        return int(duplicates)

    def _apply_dataset_specific_rules(
        self, df: pd.DataFrame, dataset_key: str
    ) -> pd.DataFrame:
        """Apply dataset specific cleaning rules based on file name hints."""

        if "accidente" in dataset_key:
            return self._clean_accidentes(df)
        if "sector" in dataset_key:
            return self._clean_sectores(df)
        return df

    def _clean_accidentes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply bespoke rules for the accidentes dataset."""

        cleaned = df.copy()

        fecha_col = self._find_first_column(
            cleaned, ["fecha_ocurrencia_acc", "fecha", "fecha_hecho"]
        )
        municipio_col = self._find_first_column(
            cleaned, ["municipio", "municipio_hecho", "municipio_accidente"]
        )
        lat_col = self._find_first_column(
            cleaned, ["latitud", "lat", "latitud_ocurrencia"]
        )
        lon_col = self._find_first_column(
            cleaned, ["longitud", "lon", "longitud_ocurrencia"]
        )

        if fecha_col:
            cleaned[fecha_col] = pd.to_datetime(
                cleaned[fecha_col], errors="coerce", dayfirst=True
            )
            min_date = pd.Timestamp("2010-01-01")
            max_date = pd.Timestamp("2026-12-31")
            date_mask = cleaned[fecha_col].between(min_date, max_date, inclusive="both")
            cleaned = cleaned.loc[date_mask]

        if municipio_col:
            municipio_series = cleaned[municipio_col].astype("string")
            cleaned[municipio_col] = municipio_series.str.strip().str.upper()

        if lat_col:
            cleaned[lat_col] = pd.to_numeric(cleaned[lat_col], errors="coerce")
        if lon_col:
            cleaned[lon_col] = pd.to_numeric(cleaned[lon_col], errors="coerce")

        if lat_col and lon_col:
            has_both = cleaned[lat_col].notna() & cleaned[lon_col].notna()
            valid_lat = cleaned[lat_col].between(-5.5, 15.5, inclusive="both")
            valid_lon = cleaned[lon_col].between(-85.0, -65.0, inclusive="both")
            valid_coords = has_both & valid_lat & valid_lon
            if municipio_col:
                keep_without_coords = ~has_both & cleaned[municipio_col].notna()
            else:
                keep_without_coords = ~has_both
            cleaned = cleaned.loc[valid_coords | keep_without_coords]
        return cleaned

    def _clean_sectores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply bespoke rules for the sectores críticos dataset."""

        cleaned = df.copy()

        text_columns = [
            "municipio",
            "departamento",
            "via",
            "tramo",
            "sentido",
        ]

        for column in text_columns:
            if column in cleaned.columns:
                series = cleaned[column].astype("string")
                normalised = series.str.strip().str.upper()
                cleaned[column] = normalised.where(~series.isna(), other=pd.NA)

        metric_keywords = ("accidente", "lesionado", "muerto", "km", "long")
        for column in cleaned.columns:
            if any(keyword in column for keyword in metric_keywords):
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

        return cleaned

    @staticmethod
    def _find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None

    def _duplicate_subset(
        self, df: pd.DataFrame, dataset_key: str
    ) -> Optional[Iterable[str]]:
        if "accidente" in dataset_key:
            subset = [
                column
                for column in [
                    "fecha_ocurrencia_acc",
                    "municipio",
                    "latitud",
                    "longitud",
                ]
                if column in df.columns
            ]
            return subset or None
        return None

    def _fill_missing_values(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Fill missing values using sensible defaults.

        Numeric columns are filled with the median while categorical columns
        are filled with the mode.
        """

        numeric_fill_values: Dict[str, float] = {}
        categorical_fill_values: Dict[str, str] = {}

        numeric_columns = df.select_dtypes(include=["number"]).columns
        for column in numeric_columns:
            series = df[column].replace({np.inf: np.nan, -np.inf: np.nan})
            median = float(series.median()) if not series.dropna().empty else 0.0
            df[column] = series.fillna(median)
            numeric_fill_values[column] = median

        categorical_columns = df.select_dtypes(exclude=["number"]).columns
        for column in categorical_columns:
            series = df[column].astype("object")
            if series.dropna().empty:
                fill_value = "desconocido"
            else:
                fill_value = series.mode(dropna=True)[0]
            df[column] = series.fillna(fill_value)
            categorical_fill_values[column] = str(fill_value)

        return numeric_fill_values, categorical_fill_values


__all__ = ["DataCleaner", "CleaningReport"]
