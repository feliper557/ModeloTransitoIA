"""
Clase base para modelos de tránsito
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseTransitoModel(ABC):
    """Clase base abstracta para modelos de tránsito"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entrenar el modelo"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hacer predicciones"""
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluar el modelo"""
        pass
    
    def save_model(self, path: str):
        """Guardar el modelo"""
        raise NotImplementedError("Método save_model debe ser implementado")
    
    def load_model(self, path: str):
        """Cargar el modelo"""
        raise NotImplementedError("Método load_model debe ser implementado")
