"""
Tests para el módulo data_loader
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Tests para la clase DataLoader"""
    
    def setUp(self):
        """Configuración antes de cada test"""
        self.data_loader = DataLoader()
    
    def test_init(self):
        """Test de inicialización"""
        self.assertIsNotNone(self.data_loader)
        self.assertIsInstance(self.data_loader.data_path, Path)
    
    def test_preprocess(self):
        """Test de preprocesamiento"""
        # Crear datos de ejemplo
        sample_data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, np.nan, 7, 8]
        })
        
        processed = self.data_loader.preprocess(sample_data)
        
        # Verificar que no hay valores nulos
        self.assertEqual(processed.isnull().sum().sum(), 0)


if __name__ == '__main__':
    unittest.main()
