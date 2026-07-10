# Analisis de Supervivencia del Titanic con Machine Learning

Proyecto academico para identificar los factores que influyeron en la supervivencia de los pasajeros del Titanic y construir un modelo predictivo con Regresion Logistica siguiendo la metodologia CRISP-ML.

## Pregunta de investigacion

Cuales fueron los factores que mas influyeron en la probabilidad de supervivencia de los pasajeros del Titanic?

## Objetivo general

Desarrollar un modelo de aprendizaje supervisado mediante Regresion Logistica para predecir la supervivencia de los pasajeros e interpretar las variables con mayor influencia.

## Modelos

- **Regresion Logistica:** predice `Survived` como variable binaria.
- **Regresion Lineal:** predice `Fare` con fines comparativos y academicos.

## Variables utilizadas

- `Pclass`: clase del pasajero.
- `Sex`: sexo.
- `Age`: edad.
- `SibSp`: hermanos o conyuge a bordo.
- `Parch`: padres o hijos a bordo.
- `Fare`: precio del boleto.
- `Embarked`: puerto de embarque.
- `FamilySize`: `SibSp + Parch + 1`.
- `IsAlone`: 1 si viaja solo, 0 si viaja con familia.

## Estructura del proyecto

```text
titanic-ml-project/
  app/
    streamlit_app.py
  data/
    titanic.csv
  landing/
    index.html
    styles.css
    script.js
    assets/
  models/
  notebooks/
    titanic_crisp_ml.ipynb
  reports/
  src/
    titanic_pipeline.py
  train_models.py
  requirements.txt
  README.md
```

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En macOS o Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entrenar los modelos

```bash
python train_models.py
```

Este comando guarda:

- `models/logistic_survival_model.joblib`
- `models/linear_fare_model.joblib`
- `reports/metrics.json`
- `reports/logistic_coefficients.csv`

## Ejecutar la aplicacion Streamlit

```bash
streamlit run app/streamlit_app.py
```

La aplicacion permite ingresar datos de un pasajero y muestra:

- Probabilidad de supervivencia.
- Prediccion: sobrevive o no sobrevive.
- Variables con mayor influencia global en el modelo.

## Abrir la landing page

Abre el archivo:

```text
landing/index.html
```

La landing resume el proyecto, los objetivos, la metodologia CRISP-ML, los modelos, resultados esperados, conclusiones, app y enlace al repositorio.

## Metodologia CRISP-ML

1. Comprension del problema.
2. Comprension de los datos.
3. Preparacion de los datos.
4. Ingenieria de caracteristicas.
5. Construccion de modelos.
6. Evaluacion.
7. Interpretacion.
8. Despliegue.
9. Conclusiones y recomendaciones.

## Publicacion en GitHub

```bash
git init
git add .
git commit -m "Proyecto Titanic ML"
git branch -M main
git remote add origin https://github.com/usuario/titanic-ml-project.git
git push -u origin main
```

Reemplaza `usuario/titanic-ml-project` por el nombre real del repositorio.

## Nota sobre imagenes RA

La landing incluye visuales locales con estetica de realidad aumentada. Si el equipo genera imagenes por IA, puede reemplazar los archivos de `landing/assets/` conservando los mismos nombres para no modificar el HTML.
