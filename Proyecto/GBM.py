# GBM.py
import os
import matplotlib
matplotlib.use("Agg")  # Para que funcione sin entorno gráfico
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 📌 Rutas relativas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "static", "images")
DATA_PATH = os.path.join(BASE_DIR, "datasheet", "dataGBM.csv")

os.makedirs(IMG_DIR, exist_ok=True)

# 📌 Cargar dataset
df = pd.read_csv(DATA_PATH)

# Variables categóricas → numéricas
label_encoders = {}
for col in ["Fuente", "sector", "Probabilidad"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 📌 Mapeo de features
FEATURE_MAP = {
    "fuente": ["Fuente"],
    "interacciones": ["Interacciones"],
    "visitas": ["Visitas"],
    "ingresos": ["Ingresos"],
    "sector": ["sector"]
}

def train_and_evaluate(feature=None):
    """
    Entrena y evalúa un modelo Gradient Boosting.
    - Si feature=None → usa todas las variables.
    - Si feature está definido → usa solo esa variable.
    """
    X = df.copy()
    y = X["Probabilidad"]

    # 🔹 Selección de variables
    if feature and feature.lower() in FEATURE_MAP:
        X = X[FEATURE_MAP[feature.lower()]]  # solo esa variable
    else:
        X = X.drop("Probabilidad", axis=1)   # todas las variables

    # 🔹 Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 🔹 Modelo GBM
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 🔹 Predicción
    y_pred = model.predict(X_test)

    # 🔹 Métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoders["Probabilidad"].classes_,
        output_dict=True
    )

    # 🔹 Importancia de las variables
    feature_importances = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": model.feature_importances_
    }).sort_values(by="Importancia", ascending=False)

    # 🔹 Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoders["Probabilidad"].classes_,
        yticklabels=label_encoders["Probabilidad"].classes_
    )
    plt.title("Matriz de Confusión - GBM")
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    # 🔹 Guardar imagen
    img_filename = f"gbm_confusion_matrix_{feature if feature else 'all'}.png"
    img_path_abs = os.path.join(IMG_DIR, img_filename)
    plt.tight_layout()
    plt.savefig(img_path_abs)
    plt.close()

    return round(accuracy, 4), report, f"images/{img_filename}", feature_importances
