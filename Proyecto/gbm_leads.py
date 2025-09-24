# gbm_leads.py
# ---------------------------------------
# Gradient Boosting Machines (GBM) aplicado a priorización de leads
# Dataset: leads.csv (ubicado en la carpeta datasheet/)
# Objetivo: Clasificar la probabilidad de conversión (Alto / Medio / Bajo)
# ---------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

# -------------------------------
# Función principal: Entrenamiento y evaluación
# -------------------------------
def evaluate(df):
    # 🔹 Normalizar nombres de columnas (evita errores con espacios)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")

    # 🔹 Separar variables (X) y objetivo (y)
    X = df.drop("ProbConversion", axis=1)
    y = df["ProbConversion"]

    # 🔹 Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 🔹 Preprocesamiento: One-Hot Encoding para variables categóricas
    categorical_features = ["Fuente", "Sector"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
        remainder="passthrough"
    )

    # 🔹 Modelo: Gradient Boosting
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])

    # 🔹 Entrenar
    model.fit(X_train, y_train)

    # 🔹 Predicciones
    y_pred = model.predict(X_test)

    # 🔹 Métricas
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # 🔹 Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de confusión - GBM")
    plt.tight_layout()
    os.makedirs("static/images", exist_ok=True)
    plt.savefig("static/images/confusion_matrix_gbm.png")
    plt.close()

    # 🔹 Retornar resultados
    return {
        "accuracy": round(acc, 4),
        "classification_report": report,
        "model": model
    }

# -------------------------------
# Función: Predicción de un lead
# -------------------------------
def predict_label(model, features):
    # features es un diccionario con los campos del dataset
    df_sample = pd.DataFrame([features])
    prediction = model.predict(df_sample)[0]
    proba = model.predict_proba(df_sample).max()
    return prediction, round(proba, 4)

# -------------------------------
# Ejecución directa por terminal
# -------------------------------
if __name__ == "__main__":
    dataset_path = "datasheet/leads.csv"

    print("Cargando dataset real:", dataset_path)
    df = pd.read_csv(dataset_path)
   
    # 🔹 Evaluar modelo
    results = evaluate(df)
    print("✅ Exactitud:", results["accuracy"])
    print(results["classification_report"])

    # 🔹 Ejemplo de predicción
    sample = {
        "Fuente": "Web",
        "Interacciones": 8,
        "Visitas": 3,
        "Ingresos": 60000,
        "Sector": "Tech"
    }
    pred, prob = predict_label(results["model"], sample)
    print(f"Predicción ejemplo: {pred} (prob={prob})")
