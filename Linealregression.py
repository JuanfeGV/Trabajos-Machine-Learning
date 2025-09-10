# Linealregression.py
import os
import pandas as pd
import numpy as np

# Usar backend "Agg" para evitar problemas en servidores/headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Datos y modelo
data = {
    "edad": [20, 25, 30, 35, 40, 45, 50, 55],
    "sueno": [8, 7, 6, 5, 7, 6, 5, 4],
    "glucosa": [85, 90, 95, 110, 100, 115, 120, 130]
}
df = pd.DataFrame(data)

X = df[["edad", "sueno"]].values
y = df["glucosa"].values

model = LinearRegression()
model.fit(X, y)

def calculateGlucose(edad: float, sueno: float) -> float:
    result = model.predict([[edad, sueno]])[0]
    return float(result)

def plot_regression():
    # Asegura carpeta static/img
    static_dir = os.path.join("static", "img")
    os.makedirs(static_dir, exist_ok=True)


    y_pred = model.predict(X)


    sueno_mean = float(df["sueno"].mean())
    edad_grid = np.linspace(df["edad"].min(), df["edad"].max(), 100)
    X_line_edad = np.column_stack([edad_grid, np.full_like(edad_grid, sueno_mean)])
    y_line_edad = model.predict(X_line_edad)

    edad_path = os.path.join(static_dir, "regresion_edad.png")
    plt.figure(figsize=(6, 4))
    plt.scatter(df["edad"], df["glucosa"], color="blue", label="Datos reales (Edad)")
    plt.plot(edad_grid, y_line_edad, color="red", label=f"Línea (sueño={sueno_mean:.1f}h)")
    plt.xlabel("Edad")
    plt.ylabel("Glucosa")
    plt.title("Edad vs Glucosa")
    plt.legend()
    plt.tight_layout()
    plt.savefig(edad_path)
    plt.close()

    # 2) Sueño vs Glucosa (mantener edad en su media para la línea)
    edad_mean = float(df["edad"].mean())
    sueno_grid = np.linspace(df["sueno"].min(), df["sueno"].max(), 100)
    X_line_sueno = np.column_stack([np.full_like(sueno_grid, edad_mean), sueno_grid])
    y_line_sueno = model.predict(X_line_sueno)

    sueno_path = os.path.join(static_dir, "regresion_sueno.png")
    plt.figure(figsize=(6, 4))
    plt.scatter(df["sueno"], df["glucosa"], color="green", label="Datos reales (Sueño)")
    plt.plot(sueno_grid, y_line_sueno, color="orange", label=f"Línea (edad={edad_mean:.0f} años)")
    plt.xlabel("Horas de sueño")
    plt.ylabel("Glucosa")
    plt.title("Sueño vs Glucosa")
    plt.legend()
    plt.tight_layout()
    plt.savefig(sueno_path)
    plt.close()

    # Devolver rutas relativas a /static
    return "img/regresion_edad.png", "img/regresion_sueno.png"
