import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Cargar dataset globalmente
data = pd.read_csv('C:\\Users\\yessi\\OneDrive\\Escritorio\\U\\Machine learning\\clone\\Trabajos-Machine-Learning\\Proyecto\\datasheet\\data1.csv')


def train_and_evaluate(features, target="Renuncia", test_size=0.2, random_state=23):
    X = data[features]
    y = data[target]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression()
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, accuracy, report, conf_matrix

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

def predict_label(model, scaler, features, threshold=0.5):
    """
    features: lista con valores [antiguedad, salario, area, horas]
    """
    X = scaler.transform([features])
    prob = model.predict_proba(X)[0][1]
    label = "Sí" if prob >= threshold else "No"
    return label, round(prob, 4)
