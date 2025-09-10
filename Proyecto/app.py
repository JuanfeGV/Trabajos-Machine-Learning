# app.py
from flask import Flask, render_template, request
import time
import Proyecto.Linealregression as Linealregression

app = Flask(__name__)

@app.route('/')  # Ruta principal
def index():
    return render_template('index.html')

# ------------------- Actividad 1 -------------------
@app.route('/actividad1')
def actividad1():
    return render_template('Actividad1/actividad1.html')

@app.route('/actividad1/caso1')
def caso1():
    return render_template('Actividad1/caso1.html')

@app.route('/actividad1/caso2')
def caso2():
    return render_template('Actividad1/caso2.html')

@app.route('/actividad1/caso3')
def caso3():
    return render_template('Actividad1/caso3.html')

@app.route('/actividad1/caso4')
def caso4():
    return render_template('Actividad1/caso4.html')

# ------------------- Actividad 2 -------------------
@app.route('/actividad2')
def actividad2():
    return render_template('Actividad2/actividad2.html')

@app.route('/actividad2/conceptosRL')
def conceptosRL():
    return render_template('Actividad2/conceptosRL.html')

@app.route('/actividad2/ejercicios', methods=["GET", "POST"])
def calculateGlucose():
    result = None
    edad_img = None
    sueno_img = None
    cache_buster = str(int(time.time()))  # Evitar caché del navegador

    if request.method == "POST":
        try:
            edad = float(request.form["edad"])
            sueno = float(request.form["sueno"])

            result = Linealregression.calculateGlucose(edad, sueno)
            edad_img, sueno_img = Linealregression.plot_regression()

        except Exception as e:
            # Muestra el error en consola para depurar rápido
            print("Error en /actividad2/ejercicios:", e)

    return render_template(
        "Actividad2/ejercicios.html",
        result=result,
        edad_img=edad_img,
        sueno_img=sueno_img,
        v=cache_buster
    )

# ------------------- Actividad 3 -------------------
@app.route('/actividad3')
def actividad3():
    return "<h1>Actividad 3 (aún no implementada)</h1>"

if __name__ == '__main__':
    app.run(debug=True)
