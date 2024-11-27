# RecomendadorRutinasEjercicio

Proyecto de IA que consta de la creación de un modelo de Redes Neuronales para recomendar rutinas de ejercicios a personas. Este es el proyecto de clase de la materia INF-556-02 (Simulación de Sistemas).

## Responsabilidades

- [ ] **Clase del Modelo:** Miguel y Daniel de la Rosa
- [ ] **Clase de la Predicción:** Jose Alberto y Luis Perez

## Instalación

### Requisitos Previos

Asegúrate de tener instaladas las siguientes herramientas:

- **Python 3.9+**: Puedes instalarlo desde [python.org](https://www.python.org/downloads/release/python-390/).

- **Pip**: Instalado automáticamente con Python, pero si no lo tienes, ejecuta:

```bash
  python -m ensurepip --upgrade
```

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/GuilleAQN/RecomendadorRutinasEjercicio.git
cd RecomendadorRutinasEjercicio
```

### Paso 2

### Con entorno virtual

Crea y activa un entorno virtual para aislar las dependencias del proyecto:

```bash
# En sistemas Unix/macOS
python3 -m venv env
source env/bin/activate

# En Windows
python -m venv env
env\Scripts\activate
```

Luego, se instalan las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

### Con [Conda](https://docs.anaconda.com/miniconda/)

Crea y activa un entorno virtual con las dependencias:

```bash
conda env create -f environment.yml
conda init
conda activate simulacion-recomendador
```

## Puntos importantes del codigo

### Procesamiento de Datos

El procesamiento de datos es una parte fundamental en la creación de modelos de Machine Learning. En este caso, se utilizó la librería `pandas` para cargar y manipular los datos.

```python
import pandas as pd

df = pd.read_csv('data/vehiculos2.csv')

# usando metodos de pandas para manipular los datos
df = df.dropna() # Elimina las filas con valores nulos
df = df.drop_duplicates() # Elimina las filas duplicadas

# Validacion de los datos
# Filtra los datos para asegurarse de que los kilometros sean válidos
df = df[
  (df["Kilometros"] >= 0)
  & ((df["Kilometros"] <= 1000) | (df["Estatus de Vehiculo"] == "Nuevo"))
]

# Normaliza el tipo de combustible, cambiando 'Gas' a 'GLP' y si no, se mantiene igual
df["Tipo Combustible"] = df["Tipo Combustible"].apply(
  lambda x: "GLP" if x == "Gas" else x
)

# Filtra los datos para incluir solo vehículos con 2 o 4 puertas
df = df[df["Numero de Puertas"].isin([2, 4])]
```

### Label Encoding

El Label Encoding es una técnica de preprocesamiento de datos que se utiliza para convertir variables categóricas en numéricas. En este caso, se utilizó para convertir las etiquetas de los ejercicios en números.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(df['exercise'])
df['exercise'] = label_encoder.transform(df['exercise'])
```

## Documentación
