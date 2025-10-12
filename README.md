# 🏢 Predicción de Ausentismo Laboral con MLOps

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-blueviolet)](https://dvc.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![Tests](https://github.com/tu-usuario/operaciones-de-aprendizaje-automatico-equipo59/actions/workflows/tests.yml/badge.svg)](https://github.com/tu-usuario/operaciones-de-aprendizaje-automatico-equipo59/actions/workflows/tests.yml)
[![Code Quality](https://github.com/tu-usuario/operaciones-de-aprendizaje-automatico-equipo59/actions/workflows/code-quality.yml/badge.svg)](https://github.com/tu-usuario/operaciones-de-aprendizaje-automatico-equipo59/actions/workflows/code-quality.yml)
[![ML Pipeline](https://github.com/tu-usuario/operaciones-de-aprendizaje-automatico-equipo59/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/tu-usuario/operaciones-de-aprendizaje-automatico-equipo59/actions/workflows/ml-pipeline.yml)

Proyecto de **Machine Learning Operations (MLOps)** desarrollado por el **Equipo 59** para predecir las horas de ausentismo laboral utilizando técnicas de regresión supervisada, con gestión de datos mediante DVC y seguimiento de experimentos con MLflow.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Dataset](#-dataset)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Metodología](#-metodología)
- [Fases del Proyecto](#-fases-del-proyecto)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Resultados](#-resultados)
- [Equipo](#-equipo)
- [Referencias](#-referencias)

---

## 🎯 Descripción del Proyecto

Este proyecto tiene como objetivo **predecir el número de horas de ausencia** de empleados de una empresa de mensajería brasileña, utilizando datos históricos de ausentismo laboral del período 2007-2010.

### **Propósito de Negocio**

- 📊 **Mejorar la planificación operativa**: anticipar ausencias para reorganizar equipos
- 💰 **Reducir costos**: minimizar impacto de ausencias inesperadas
- 🏥 **Diseñar políticas de bienestar**: identificar patrones de salud y ausentismo
- 🎯 **Optimizar recursos humanos**: tomar decisiones estratégicas basadas en datos

### **Stakeholders**

- **RRHH**: diseño de programas de salud y reducción de ausentismo
- **Supervisores**: organización de equipos y cargas de trabajo
- **Dirección**: decisiones estratégicas sobre contratación y retención
- **Empleados**: beneficiarios indirectos de políticas personalizadas

---

## 📊 Dataset

**Fuente**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work) - Absenteeism at Work

### **Características del Dataset**

- **Período**: 2007-2010
- **Instancias**: 754 registros
- **Variables**: 20 features + 1 target
- **Tipo de problema**: Regresión supervisada
- **Variable objetivo**: `Absenteeism time in hours`

### **Categorías de Variables**

#### 🔹 Variables Demográficas
- `Age`: Edad del empleado
- `Education`: Nivel educativo (1=secundaria, 2=licenciatura, 3=posgrado, 4=maestría/doctorado)
- `Son`: Número de hijos

#### 🔹 Variables Laborales
- `Service time`: Antigüedad en la empresa
- `Work load Average/day`: Carga de trabajo promedio diaria
- `Disciplinary failure`: Registro de faltas disciplinarias (0/1)
- `Hit target`: Nivel de cumplimiento de objetivos

#### 🔹 Variables de Salud y Estilo de Vida
- `Reason for absence`: Código ICD (International Classification of Diseases) - 29 categorías
- `Social drinker`: Consumo social de alcohol (0/1)
- `Social smoker`: Fumador social (0/1)
- `Weight`: Peso en kg
- `Height`: Altura en cm
- `Body mass index`: Índice de masa corporal

#### 🔹 Variables Contextuales
- `Transportation expense`: Gasto en transporte
- `Distance from Residence to Work`: Distancia al trabajo en km
- `Seasons`: Estación del año (1=verano, 2=otoño, 3=invierno, 4=primavera)
- `Month of absence`: Mes de la ausencia (0-12)
- `Day of the week`: Día de la semana (2-6, lunes-viernes)
- `Pet`: Número de mascotas

---

## 📁 Estructura del Repositorio

```
operaciones-de-aprendizaje-automatico-equipo59/
│
├── AbsenteeismAtWork/                      # Proyecto principal
│   ├── data/                               # Datos versionados con DVC
│   │   ├── work_absenteeism_original.csv.dvc
│   │   ├── work_absenteeism_clean.csv.dvc
│   │   └── work_absenteeism_modified.csv
│   │
│   ├── Phase-1/                            # Notebooks Fase 1
│   │   ├── data_preparation.ipynb          # Pipeline de limpieza
│   │   ├── eda_fe.ipynb                    # EDA y Feature Engineering
│   │   └── work_absenteeism.csv
│   │
│   ├── mlruns/                             # Experimentos MLflow
│   │   ├── 0/                              # Experimento default
│   │   └── models/                         # Modelos registrados
│   │
│   └── phase_1.ipynb                       # ML Canvas consolidado
│
├── scripts/                                # Scripts de apoyo
│   ├── 1.intro_mlflow.ipynb               # Tutorial MLflow
│   └── Fase1.ipynb                        # Pipeline completo
│
├── README.md                               # Este archivo
├── LICENSE                                 # Licencia del proyecto
└── links.txt                              # Referencias útiles
```

---

## 🚀 Instalación

### **Prerequisitos**

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git

### **1. Clonar el repositorio**

```bash
git clone https://github.com/tu-usuario/operaciones-de-aprendizaje-automatico-equipo59.git
cd operaciones-de-aprendizaje-automatico-equipo59
```

### **2. Crear entorno virtual (recomendado)**

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **3. Instalar dependencias**

```bash
pip install -r requirements.txt
```

O instalar manualmente:

```bash
pip install mlflow
pip install pandas numpy scipy scikit-learn
pip install matplotlib seaborn
pip install ydata-profiling
pip install jupyter notebook
pip install pytest  # Para testing
```

### **4. Configurar DVC (opcional, para versionado de datos)**

```bash
pip install dvc
dvc pull  # Descarga los datasets versionados
```

---

## 🔄 CI/CD Pipeline

Este proyecto incluye una suite completa de **GitHub Actions workflows** para automatización:

### **Workflows Activos**

| Workflow | Trigger | Descripción |
|----------|---------|-------------|
| **Tests** | Push/PR | Ejecuta tests unitarios en múltiples plataformas y versiones de Python |
| **Code Quality** | Push/PR | Linting, formateo, type checking y análisis de seguridad |
| **ML Pipeline** | Cambios en datos/notebooks | Valida pipeline de preprocesamiento y notebooks |
| **Documentation** | Cambios en markdown | Valida documentación y enlaces |
| **DVC Sync** | Cambios en .dvc | Valida versionado de datos |
| **Model Training** | Push a main/Manual | Entrena modelo baseline y valida performance |

### **Estado de los Workflows**

Puedes ver el estado actual en la pestaña [Actions](../../actions) del repositorio.

---

## 💻 Uso

### **Iniciar MLflow UI**

Para visualizar experimentos y métricas:

```bash
cd AbsenteeismAtWork
mlflow ui
```

Abre tu navegador en: `http://127.0.0.1:5000`

### **Ejecutar notebooks**

```bash
jupyter notebook
```

Navega a los notebooks en el orden sugerido:

1. `AbsenteeismAtWork/Phase-1/data_preparation.ipynb` - Limpieza de datos
2. `AbsenteeismAtWork/Phase-1/eda_fe.ipynb` - Análisis exploratorio
3. `scripts/Fase1.ipynb` - Pipeline completo

### **Pipeline de Preprocesamiento**

El pipeline incluye las siguientes etapas:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

preprocessing_pipeline = Pipeline([
    ('drop_columns', FunctionTransformer(drop_columns)),
    ('strip_objects', FunctionTransformer(strip_object_columns)),
    ('safe_round', FunctionTransformer(safe_round_to_int_df)),
    ('fix_invalids', FunctionTransformer(fix_invalid_values)),
    ('winsorize', FunctionTransformer(winsorize_iqr)),
    ('fillna', FunctionTransformer(fillna_with_median)),
    ('final_int', FunctionTransformer(final_int_conversion))
])

df_clean = preprocessing_pipeline.fit_transform(df)
```

### **Ejecutar Tests**

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Con cobertura de código
pytest tests/ --cov=. --cov-report=html

# Ver reporte de cobertura
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

### **Validar Calidad de Código**

```bash
# Linting
flake8 .

# Formateo automático
black .
isort .

# Análisis de seguridad
bandit -r .
```

---

## 🔬 Metodología

### **Machine Learning Canvas**

El proyecto sigue la metodología [ML Canvas](https://ignaciogavilan.com/metodologia-para-machine-learning-ii-machine-learning-canvas/) que estructura el desarrollo en 4 etapas:

1. **Objetivo** - Definición del problema y stakeholders
2. **Aprender** - Recolección y análisis de datos
3. **Predecir** - Selección de algoritmos y métricas
4. **Evaluar** - Monitorización y mejora continua

### **Pipeline de Preprocesamiento**

#### 🗑️ Variables Eliminadas

| Variable | Razón |
|----------|-------|
| `ID` | Identificador único sin valor predictivo |
| `mixed_type_col` | Columna inconsistente sin información útil |

#### 🔠 Transformaciones en Variables Categóricas

| Variable | Transformación | Justificación |
|----------|---------------|---------------|
| `Reason for absence` | Valores fuera de 0-28 → 0 | Mantener consistencia con códigos ICD |
| `Month of absence` | Valores fuera de 0-12 → 0 | Asegurar meses válidos |
| `Day of the week` | Valores fuera de 2-6 → moda | Solo días laborales |
| `Seasons` | Valores fuera de 1-4 → moda | 4 estaciones válidas |
| Variables binarias | Valores != 0 o 1 → moda | Mantener integridad binaria |

#### 🔢 Transformaciones en Variables Numéricas

Para **12 variables numéricas** se aplicó:

1. **Winsorización IQR**: Control de outliers
2. **Imputación con mediana**: Relleno de valores nulos
3. **Redondeo final**: Conversión a enteros

**Resultado**: 
- ✅ 0 valores nulos
- ✅ Todas las variables en formato `int64`
- ✅ Outliers controlados

### **Análisis Exploratorio**

#### Correlaciones con el Target

| Variable | Correlación |
|----------|-------------|
| `Disciplinary failure` | -0.263 (negativa fuerte) |
| `Reason for absence` | -0.167 |
| `Day of the week` | -0.114 |
| `Son` | +0.122 |
| `Transportation expense` | +0.121 |
| `Social drinker` | +0.117 |

**Insight**: Las faltas disciplinarias son el predictor más fuerte (correlación negativa).

---

## 📈 Fases del Proyecto

### ✅ **Fase 1: Preparación y Análisis** (Completada)

- [x] Definición del problema (ML Canvas)
- [x] Carga y exploración inicial del dataset
- [x] Pipeline de limpieza de datos
- [x] Análisis exploratorio de datos (EDA)
- [x] Feature Engineering básico
- [x] Versionado de datos con DVC
- [x] Setup de MLflow

### 🔄 **Fase 2: Modelado** (En progreso)

- [ ] Entrenamiento de modelos baseline
- [ ] Experimentación con múltiples algoritmos:
  - Regresión Lineal
  - Random Forest Regressor
  - Gradient Boosting (XGBoost, LightGBM)
  - CatBoost
- [ ] Optimización de hiperparámetros
- [ ] Validación cruzada
- [ ] Registro de experimentos en MLflow

### ⏳ **Fase 3: Evaluación y Selección**

- [ ] Comparación de métricas (MAE, RMSE, R²)
- [ ] Análisis de importancia de features
- [ ] Selección del mejor modelo
- [ ] Evaluación en conjunto de prueba
- [ ] Análisis de residuos

### ⏳ **Fase 4: Despliegue y Monitoreo**

- [ ] Empaquetado del modelo final
- [ ] Desarrollo de API (FastAPI/Flask)
- [ ] Containerización con Docker
- [ ] Monitoreo de drift
- [ ] Documentación de producción

---

## 🛠️ Tecnologías Utilizadas

### **Lenguajes y Librerías**

- **Python 3.8+**: Lenguaje principal
- **Pandas**: Manipulación de datos
- **NumPy**: Cálculos numéricos
- **Scikit-learn**: Modelado y preprocesamiento
- **Matplotlib & Seaborn**: Visualización

### **MLOps Tools**

- **MLflow**: Tracking de experimentos y registro de modelos
- **DVC**: Control de versiones de datos
- **Jupyter Notebook**: Desarrollo interactivo

### **Herramientas de Análisis**

- **ydata-profiling**: Generación de reportes automáticos
- **SciPy**: Análisis estadístico

---

## 📊 Resultados

### **Versionado de Datos**

| Versión | Archivo | Descripción | Tamaño | Fecha |
|---------|---------|-------------|--------|-------|
| v1 | `work_absenteeism_original.csv` | Dataset original sin procesar | 45 KB | 2025-10-09 |
| v2 | `work_absenteeism_clean.csv` | Dataset limpio y procesado | 41 KB | 2025-10-10 |

### **Feature Engineering**

- **Normalización**: StandardScaler aplicado a 8 variables continuas
- **Encoding**: One-hot encoding de variables categóricas
- **PCA**: 2 componentes principales explican **51.88%** de la varianza

### **Métricas Objetivo**

Para la fase de modelado se utilizarán:

- **MAE** (Mean Absolute Error): Error promedio en horas
- **RMSE** (Root Mean Squared Error): Penaliza errores grandes
- **R²** (Coeficiente de Determinación): Proporción de varianza explicada
- **Validación Cruzada**: K-Fold (k=5)

---

## 👥 Equipo

**Equipo 59** - Master en Inteligencia Artificial

- Jorge - Data Preprocessing & Feature Engineering
- Miguel - EDA & Data Analysis
- [Añadir más miembros según corresponda]

**Institución**: Tecnológico de Monterrey  
**Curso**: Operaciones de Aprendizaje Automático

---

## 📚 Referencias

### **Dataset**
- [UCI ML Repository - Absenteeism at Work Dataset](https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work)

### **Metodología**
- [Machine Learning Canvas - Ignacio Gavilán](https://ignaciogavilan.com/metodologia-para-machine-learning-ii-machine-learning-canvas/)

### **Herramientas**
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### **Papers y Artículos**
- International Classification of Diseases (ICD) - WHO

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

<div align="center">

Desarrollado con ❤️ por el Equipo 59

</div>
