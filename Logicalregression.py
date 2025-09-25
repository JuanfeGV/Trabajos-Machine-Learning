import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Cargar dataset globalmente
data = pd.read_csv('datasheet\data1.csv')


def evaluate(features, target="Renuncia", test_size=0.2, random_state=23):
    X = data[features].copy()
    y = data[target]

    # Definir columnas categóricas y numéricas
    categorical_features = [col for col in features if col == "Areadetrabajo"]
    numeric_features = [col for col in features if col != "Areadetrabajo"]

    # Transformador: OneHotEncoder para categóricas, StandardScaler para numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features)
        ],
        remainder="drop"
    )

    # División train/test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocesamiento
    x_train_pre = preprocessor.fit_transform(x_train)
    x_test_pre = preprocessor.transform(x_test)

    # Modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_pre, y_train)

    # Predicciones
    y_pred = model.predict(x_test_pre)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, accuracy*100, report, conf_matrix


def save_confusion_matrix(conf_matrix, filename="matriz.png", variable=None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(BASE_DIR, "static", "images")
    os.makedirs(static_dir, exist_ok=True)
    path = os.path.join(static_dir, filename)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel(f"Predicted - {variable}" if variable else "Predicted")
    plt.ylabel("Actual")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return f"images/{filename}".replace("\\", "/")


def predict_label(model, preprocessor, features, threshold=0.5):
    """
    features: lista con valores [antiguedad, salario, area, horas]
    """
    X = pd.DataFrame([features], columns=["Antiguedad", "Nivelsalarial(smlv)", "Areadetrabajo", "Horasextra"])
    X_pre = preprocessor.transform(X)
    prob = model.predict_proba(X_pre)[0][1]
    label = "Sí" if prob >= threshold else "No"
    return label, round(prob, 4)
