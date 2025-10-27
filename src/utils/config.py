"""
Gestión de configuración del proyecto
"""

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv


class Config:
    """Clase para gestionar la configuración del proyecto"""
    
    def __init__(self, env_file: str = ".env"):
        self.root_path = Path(__file__).parent.parent.parent
        self.env_file = self.root_path / env_file
        
        # Cargar variables de entorno
        if self.env_file.exists():
            load_dotenv(self.env_file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor de configuración"""
        return os.getenv(key, default)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Obtener valor booleano de configuración"""
        value = self.get(key, str(default))
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Obtener valor entero de configuración"""
        try:
            return int(self.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Obtener valor flotante de configuración"""
        try:
            return float(self.get(key, default))
        except (ValueError, TypeError):
            return default
    
    @property
    def debug(self) -> bool:
        """Modo debug activado"""
        return self.get_bool('DEBUG', False)
    
    @property
    def data_path(self) -> Path:
        """Ruta de datos"""
        return self.root_path / self.get('DATA_PATH', './data')
    
    @property
    def model_path(self) -> Path:
        """Ruta de modelos"""
        return self.root_path / self.get('MODEL_PATH', './models')
    
    @property
    def output_path(self) -> Path:
        """Ruta de salida"""
        return self.root_path / self.get('OUTPUT_PATH', './output')
