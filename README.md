# ModeloTransitoIA

Un proyecto de Python para modelos de tránsito con Inteligencia Artificial.

## Descripción

Este proyecto implementa modelos de IA para análisis y predicción de tránsito.

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
python main.py
```

## Estructura del Proyecto

```
ModeloTransitoIA/
│
├── src/                 # Código fuente
│   ├── __init__.py
│   ├── models/         # Modelos de IA
│   ├── data/           # Procesamiento de datos
│   └── utils/          # Utilidades
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
