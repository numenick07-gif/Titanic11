from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


current = Path(__file__).resolve().parent
if (current / "src").exists():
    PROJECT_ROOT = current
else:
    PROJECT_ROOT = current.parent
DATA_PATH = PROJECT_ROOT / "data" / "titanic.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["FamilySize"] = prepared["SibSp"].fillna(0) + prepared["Parch"].fillna(0) + 1
    prepared["IsAlone"] = (prepared["FamilySize"] == 1).astype(int)
    return prepared


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_logistic_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def build_linear_model() -> Pipeline:
    regression_numeric = ["Pclass", "Age", "SibSp", "Parch", "FamilySize", "IsAlone"]
    regression_categorical = ["Sex", "Embarked", "Survived"]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, regression_numeric),
            ("cat", categorical_pipeline, regression_categorical),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    coefficients = model.named_steps["model"].coef_[0]
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        .assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def train_and_evaluate(data_path: Path = DATA_PATH) -> dict:
    df = prepare_features(load_data(data_path))

    X = df[FEATURES]
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logistic_model = build_logistic_model()
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    y_proba = logistic_model.predict_proba(X_test)[:, 1]

    classification_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    regression_df = df.dropna(subset=["Fare"])
    X_reg = regression_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Survived", "FamilySize", "IsAlone"]]
    y_reg = regression_df["Fare"]
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    linear_model = build_linear_model()
    linear_model.fit(X_reg_train, y_reg_train)
    fare_pred = linear_model.predict(X_reg_test)
    mse = mean_squared_error(y_reg_test, fare_pred)
    regression_metrics = {
        "mae": float(mean_absolute_error(y_reg_test, fare_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_reg_test, fare_pred)),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic_model, MODELS_DIR / "logistic_survival_model.joblib")
    joblib.dump(linear_model, MODELS_DIR / "linear_fare_model.joblib")

    coefficients = logistic_coefficients(logistic_model)
    coefficients.to_csv(REPORTS_DIR / "logistic_coefficients.csv", index=False)

    return {
        "classification": classification_metrics,
        "regression": regression_metrics,
        "coefficients": coefficients,
        "rows": int(len(df)),
    }
