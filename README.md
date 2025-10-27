# ModeloTransitoIA

Un proyecto de Python para modelos de tránsito con Inteligencia Artificial.

## Descripción

El proyecto está organizado en tres etapas principales, todas implementadas en Python:

1. **Limpieza de datos:** se normalizan nombres de columnas, se eliminan duplicados y se imputan valores faltantes.
2. **Visualización:** se generan gráficas descriptivas (histogramas, matrices de correlación y series de tiempo) para explorar los datos procesados.
3. **Entrenamiento con aprendizaje por refuerzo:** se entrena un agente Q-learning sencillo que ajusta un semáforo virtual utilizando las métricas del conjunto de datos.

## Instalación

1. Crear un entorno virtual:
```bash
python -m venv venv
```

2. Activar el entorno virtual:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py --input-file tips.csv --metric-column total_bill --plot-prefix demo
```

Parámetros opcionales:

- `--input-file`: archivo CSV presente en la carpeta `data/` (por defecto `tips.csv`).
- `--metric-column`: columna numérica que se utilizará como métrica para el entrenamiento de refuerzo.
- `--plot-prefix`: prefijo que se añadirá a los archivos PNG generados por la etapa de visualización.

## Estructura del Proyecto

```
ModeloTransitoIA/
│
├── src/                 # Código fuente
│   ├── __init__.py
│   ├── models/         # Modelos de IA (incluye refuerzo)
│   ├── data/           # Procesamiento y limpieza de datos
│   ├── pipeline/       # Orquestación de las tres etapas
│   ├── utils/          # Utilidades (configuración, logging)
│   └── visualization/  # Generación de gráficas
│
├── tests/              # Tests unitarios
├── data/               # Datos de entrada
├── notebooks/          # Jupyter notebooks
├── requirements.txt    # Dependencias
└── main.py            # Punto de entrada
```

## Control de Versiones

Este proyecto utiliza Git para control de versiones.

### Comandos básicos:

```bash
# Ver estado de cambios
git status

# Agregar archivos al staging
git add .

# Hacer commit
git commit -m "Descripción del cambio"

# Ver historial
git log --oneline
```

## Licencia

MIT License
