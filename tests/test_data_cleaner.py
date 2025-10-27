"""Tests for the DataCleaner component."""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.data_cleaner import DataCleaner


def test_data_cleaner_normalises_columns_and_fills_missing():
    cleaner = DataCleaner()
    raw = pd.DataFrame(
        {
            "Vehículos": [10, 12, None, 12],
            "Estado": ["Alto", None, "Bajo", "Bajo"],
        }
    )

    cleaned, report = cleaner.clean(raw)

    assert list(cleaned.columns) == ["vehículos", "estado"]
    assert cleaned.isna().sum().sum() == 0
    # Dos filas quedan duplicadas después de la imputación
    assert report.as_dict()["duplicates_removed"] == 2


def test_clean_accidentes_applies_domain_rules():
    cleaner = DataCleaner()
    raw = pd.DataFrame(
        {
            "FECHA_OCURRENCIA_ACC": [
                "15/01/2015",
                "2005-01-01",
                "2025-05-05",
                "2018-03-03",
            ],
            "MUNICIPIO": ["Cali", "Medellín", "Bogotá", None],
            "LATITUD": ["3.45", "4.56", "20", None],
            "LONGITUD": ["-76.5", "-90", "-74.1", None],
        }
    )

    cleaned, report = cleaner.clean(raw, dataset="ACCIDENTE.csv")

    assert "fecha_ocurrencia_acc" in cleaned.columns
    assert pd.api.types.is_datetime64_dtype(cleaned["fecha_ocurrencia_acc"])
    # Solo la primera fila cumple con fecha válida y coordenadas plausibles
    assert len(cleaned) == 1
    row = cleaned.iloc[0]
    assert row["municipio"] == "CALI"
    assert row["latitud"] == 3.45
    assert row["longitud"] == -76.5
    assert report.as_dict()["duplicates_removed"] == 0


def test_clean_sectores_coerces_text_and_numeric_fields():
    cleaner = DataCleaner()
    raw = pd.DataFrame(
        {
            "Municipio": ["  bogotá ", None],
            "Departamento": ["Cundinamarca", "cundinamarca"],
            "Via": ["av. 26", "calle 13"],
            "Tramo": [" norte ", "Sur"],
            "Sentido": ["oriente", "occidente"],
            "Accidentes": ["10", " 5 "],
            "Longitud (km)": ["1.5", "dos"],
        }
    )

    cleaned, report = cleaner.clean(
        raw, dataset="SECTORES_CRITICOS_DE_SINIESTRALIDAD_VIAL.csv"
    )

    for column in ["municipio", "departamento", "via", "tramo", "sentido"]:
        assert cleaned[column].iloc[0] == cleaned[column].iloc[0].upper()

    assert cleaned["accidentes"].tolist() == [10.0, 5.0]
    # Las columnas métricas quedan en formato numérico y con imputación coherente
    assert cleaned["longitud_km"].tolist() == [1.5, 1.5]
    assert report.as_dict()["numeric_filled"]["longitud_km"] == 1.5
