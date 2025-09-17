import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Cargar el conjunto de datos
data = pd.read_csv('C:\\Users\\yessi\\OneDrive\\Escritorio\\U\\Machine learning\\Proyectos\\Clase 4\\Proyecto\\datasheet\\data.csv')

#Explorar el conjunto de datos
print(data.head())
print(data.info())
print(data.describe())

#separar las variables independientes (X) y la variable dependiente (y)
x = data.drop('HeartDisease', axis=1) #Supón que 'target' es la columna de etiqueta/clase'
y = data['HeartDisease']

# Dividir el conjunto de datos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Estandarizar los datos (opcional pero recomendado para regresión logística)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Crear el modelo de regresión logística
logistic_model = LogisticRegression()

#Entrenar el modelo
logistic_model.fit(x_train_scaled, y_train)

#Realizar predicciones con el conjunto de prueba
y_pred = logistic_model.predict(x_test_scaled)

#Crear la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

#Visualizar la matriz de confusión



def logical_regression():
    import os
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
    static_dir = os.path.join(BASE_DIR, "static", "images")
    os.makedirs(static_dir, exist_ok=True)
    logic_path = os.path.join(static_dir, "regresion_logica.png")
    
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(logic_path)
    plt.close()

    # Devuelvo la ruta relativa para Flask
    return "images/regresion_logica.png"

logical_regression()

# Reporte de clasificación y exactitud
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud del modelo: {accuracy:.2f}%')