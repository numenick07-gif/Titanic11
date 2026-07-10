from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

current = Path(__file__).resolve().parent
if (current / "src").exists():
    PROJECT_ROOT = current
else:
    PROJECT_ROOT = current.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.titanic_pipeline import MODELS_DIR, logistic_coefficients, train_and_evaluate


MODEL_PATH = MODELS_DIR / "logistic_survival_model.joblib"


@st.cache_resource
def load_or_train_model():
    if not MODEL_PATH.exists():
        train_and_evaluate()
    return joblib.load(MODEL_PATH)


def build_passenger_frame(
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
st.caption("Modelo de Regresion Logistica basado en variables del pasajero.")

model = load_or_train_model()

left, right = st.columns([0.9, 1.1], gap="large")

with left:
    st.subheader("Datos del pasajero")
    pclass = st.selectbox("Clase del pasajero", [1, 2, 3], index=2)
    sex_label = st.selectbox("Sexo", ["Mujer", "Hombre"])
    age = st.slider("Edad", 0.0, 80.0, 29.0, 1.0)
    fare = st.number_input("Tarifa del boleto", min_value=0.0, value=32.0, step=1.0)
    sibsp = st.number_input("Hermanos o conyuge a bordo", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Padres o hijos a bordo", min_value=0, max_value=10, value=0, step=1)
    embarked_label = st.selectbox("Puerto de embarque", ["Southampton", "Cherbourg", "Queenstown"])

sex = "female" if sex_label == "Mujer" else "male"
embarked_map = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}
embarked = embarked_map[embarked_label]
passenger = build_passenger_frame(pclass, sex, age, sibsp, parch, fare, embarked)

probability = float(model.predict_proba(passenger)[0, 1])
prediction = int(probability >= 0.5)

with right:
    st.subheader("Resultado")
    st.metric("Probabilidad de supervivencia", f"{probability:.1%}")
    st.metric("Prediccion", "Sobrevive" if prediction else "No sobrevive")
    st.progress(probability)

    st.subheader("Variables consideradas")
    display_passenger = passenger.rename(
        columns={
            "Pclass": "Clase",
            "Sex": "Sexo",
            "Age": "Edad",
            "SibSp": "Hermanos/Conyuge",
            "Parch": "Padres/Hijos",
            "Fare": "Tarifa",
            "Embarked": "Embarque",
            "FamilySize": "Tamano familia",
            "IsAlone": "Viaja solo",
        }
    )
    st.dataframe(display_passenger, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Factores de mayor influencia global")
coef = logistic_coefficients(model).head(10)
coef["impacto"] = coef["coefficient"].apply(lambda value: "Aumenta supervivencia" if value > 0 else "Reduce supervivencia")
st.dataframe(coef[["feature", "coefficient", "impacto"]], use_container_width=True, hide_index=True)

st.caption(
    "La interpretacion se basa en los coeficientes del modelo. "
    "Valores positivos aumentan la probabilidad estimada y valores negativos la reducen."
)
