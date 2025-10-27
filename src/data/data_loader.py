"""
Cargador de datos para el proyecto
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """Clase para cargar y preprocesar datos de tránsito"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.data = None
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Cargar datos desde un archivo CSV
        
        Args:
            filename: Nombre del archivo CSV
            
        Returns:
            DataFrame con los datos cargados
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        self.data = pd.read_csv(file_path)
        print(f"Datos cargados: {self.data.shape[0]} filas, {self.data.shape[1]} columnas")
        
        return self.data
    
    def preprocess(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocesar los datos
        
        Args:
            data: DataFrame a preprocesar (usa self.data si es None)
            
        Returns:
            DataFrame preprocesado
        """
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No hay datos para preprocesar")
        
        # TODO: Implementar lógica de preprocesamiento específica
        processed_data = data.copy()
        
        # Ejemplo: Eliminar valores nulos
        processed_data = processed_data.dropna()
        
        return processed_data
    
    def split_data(
        self, 
        data: pd.DataFrame, 
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Dividir datos en conjuntos de entrenamiento y prueba
        
        Args:
            data: DataFrame con los datos
            target_column: Nombre de la columna objetivo
            test_size: Proporción del conjunto de prueba
            random_state: Semilla para reproducibilidad
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X = data.drop(columns=[target_column]).values
        y = data[target_column].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
