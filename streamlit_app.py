from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def find_data_file() -> Path | None:
    current = Path(__file__).resolve()
    candidates = [
        current.parent / "data" / "titanic.csv",
        current.parent.parent / "data" / "titanic.csv",
        current.parent / "titanic.csv",
        current.parent.parent / "titanic.csv",
        Path("data") / "titanic.csv",
        Path("titanic.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = find_data_file()
    if data_path is None:
        st.error(
            "No se encontro el archivo titanic.csv. "
            "Sube el dataset en una carpeta llamada data/ o en la misma carpeta de la app."
        )
        st.stop()
    return pd.read_csv(data_path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["FamilySize"] = data["SibSp"].fillna(0) + data["Parch"].fillna(0) + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    return data


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


@st.cache_resource
def train_model() -> tuple[Pipeline, dict, pd.DataFrame]:
    df = add_features(load_data())
    X = df[FEATURES]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    coefficients = get_coefficients(model)
    return model, metrics, coefficients


def get_coefficients(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    values = model.named_steps["model"].coef_[0]
    return (
        pd.DataFrame({"Variable": feature_names, "Coeficiente": values})
        .assign(Importancia=lambda data: data["Coeficiente"].abs())
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )


def passenger_dataframe(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
) -> pd.DataFrame:
    family_size = sibsp + parch + 1
    return pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked,
                "FamilySize": family_size,
                "IsAlone": int(family_size == 1),
            }
        ]
    )


st.set_page_config(page_title="Titanic ML", page_icon="T", layout="wide")

st.title("Prediccion de Supervivencia en el Titanic")
st.write(
    "Aplicacion interactiva con Regresion Logistica para estimar la probabilidad "
    "de supervivencia de un pasajero."
)

model, metrics, coefficients = train_model()

st.sidebar.header("Datos del pasajero")
pclass = st.sidebar.selectbox("Clase del pasajero", [1, 2, 3], index=2)
sex_label = st.sidebar.selectbox("Sexo", ["Mujer", "Hombre"])
age = st.sidebar.slider("Edad", 0, 80, 29)
fare = st.sidebar.number_input("Tarifa del boleto", min_value=0.0, value=32.0, step=1.0)
sibsp = st.sidebar.number_input("Hermanos o conyuge a bordo", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("Padres o hijos a bordo", min_value=0, max_value=10, value=0)
embarked_label = st.sidebar.selectbox("Puerto de embarque", ["Southampton", "Cherbourg", "Queenstown"])

sex = "female" if sex_label == "Mujer" else "male"
embarked = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}[embarked_label]
passenger = passenger_dataframe(pclass, sex, age, sibsp, parch, fare, embarked)

probability = float(model.predict_proba(passenger)[0, 1])
prediction = "Sobrevive" if probability >= 0.5 else "No sobrevive"

result_col, data_col = st.columns([1, 1])

with result_col:
    st.subheader("Resultado del modelo")
    st.metric("Probabilidad de supervivencia", f"{probability:.1%}")
    st.metric("Prediccion", prediction)
    st.progress(probability)

with data_col:
    st.subheader("Datos ingresados")
    st.dataframe(passenger, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Metricas del modelo")
metric_cols = st.columns(5)
metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
metric_cols[1].metric("Precision", f"{metrics['precision']:.1%}")
metric_cols[2].metric("Recall", f"{metrics['recall']:.1%}")
metric_cols[3].metric("F1-score", f"{metrics['f1']:.1%}")
metric_cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.1%}")

st.subheader("Matriz de confusion")
cm = metrics["confusion_matrix"]
cm_df = pd.DataFrame(
    cm,
    index=["Real: No sobrevivio", "Real: Sobrevivio"],
    columns=["Predicho: No sobrevivio", "Predicho: Sobrevivio"],
)
st.dataframe(cm_df, use_container_width=True)

st.subheader("Variables con mayor influencia")
top_coefficients = coefficients.head(10).copy()
top_coefficients["Efecto"] = np.where(
    top_coefficients["Coeficiente"] > 0,
    "Aumenta supervivencia",
    "Reduce supervivencia",
)
st.dataframe(
    top_coefficients[["Variable", "Coeficiente", "Efecto"]],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Valores positivos aumentan la probabilidad estimada de supervivencia. "
    "Valores negativos la reducen."
)
