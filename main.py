"""
ModeloTransitoIA - Main Entry Point
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logger


def main():
    """Función principal del proyecto"""
    logger = setup_logger()
    
    logger.info("Iniciando ModeloTransitoIA...")
    
    # TODO: Implementar la lógica principal aquí
    print("¡Bienvenido a ModeloTransitoIA!")
    print("Este es el punto de entrada de tu aplicación.")
    
    logger.info("Proceso completado exitosamente.")


if __name__ == "__main__":
    main()
