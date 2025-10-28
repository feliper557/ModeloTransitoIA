"""ModeloTransitoIA - Main entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import TrafficWorkflow
from src.utils.config import Config
from src.utils.logger import setup_logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta el flujo completo: limpieza, visualización y entrenamiento RL.",
    )
    parser.add_argument(
        "--input-file",
        default="tips.csv",
        help="Nombre del archivo CSV localizado en la carpeta de datos.",
    )
    parser.add_argument(
        "--metric-column",
        default=None,
        help="Columna que se usará como métrica principal para el entrenamiento RL.",
    )
    parser.add_argument(
        "--plot-prefix",
        default=None,
        help="Prefijo para nombrar los archivos de salida de las gráficas.",
    )
    return parser.parse_args()


def main() -> None:
    """Función principal del proyecto."""

    args = parse_arguments()
    logger = setup_logger()
    config = Config()

    workflow = TrafficWorkflow(config=config, logger=logger)

    try:
        result = workflow.run(
            input_file=args.input_file,
            metric_column=args.metric_column,
            plot_prefix=args.plot_prefix,
        )
    except FileNotFoundError as exc:
        logger.error("No se encontró el archivo especificado: %s", exc)
        raise SystemExit(1) from exc
    except ValueError as exc:
        logger.error("Error al ejecutar el flujo: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Flujo completado. Recompensa media: %.2f", sum(result["training"]["episode_rewards"]) / len(result["training"]["episode_rewards"]))

    print("\nResumen del flujo ejecutado")
    print("---------------------------")
    print(f"Archivo limpiado: {args.input_file}")
    print(f"Filas resultantes: {result['cleaning_report']['final_rows']}")
    print(f"Gráficas generadas: {len(result['plots'])}")
    print(
        "Columna utilizada para RL:",
        result["training"]["metric_column"],
    )
    print("Primeros 5 valores de la política aprendida:", result["training"]["policy"][:5])

    if result.get("comparison"):
        comparison = result["comparison"]
        best_neural = comparison.get("neural_best")
        if best_neural:
            print(
                "Mejor red neuronal (MSE más bajo):",
                best_neural["name"],
                "con MSE =",
                round(best_neural["metrics"]["mse"], 4),
            )
        print(
            "Recompensa media del modelo de refuerzo:",
            round(comparison.get("reinforcement_average_reward", 0.0), 3),
        )


if __name__ == "__main__":
    main()
