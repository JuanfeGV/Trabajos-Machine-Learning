# app.py
from flask import Flask, render_template, request
import time
import Linealregression as Linealregression
from Logicalregression import train_and_evaluate, save_confusion_matrix

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

@app.route('/actividad2/glucosa')
def glucosa():
    return render_template('glucosa.html')

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
    return render_template('Actividad3/actividad3.html')

@app.route('/actividad3/conceptosRLog')
def conceptosRLog():
    return render_template('Actividad3/conceptosRLog.html')


@app.route('/actividad3/ejercicios')
def logicalregression():
    variable = request.args.get("var", "horas_extra")  # valor por defecto

    valid_features = {
        "antiguedad": ["Antiguedad"],
        "salario": ["Nivelsalarial(smlv)"],
        "horas_extra": ["Horasextra"],
        "area": ["Areadetrabajo"]
    }

    features = valid_features[variable]

    try:
        model, accuracy, report, conf_matrix = train_and_evaluate(features)
        img_path = save_confusion_matrix(conf_matrix, variable)
    except Exception as e:
        # Mostrar el error real en la página
        import traceback
        return f"<h1>Error al procesar '{variable}'</h1><pre>{traceback.format_exc()}</pre>"

    return render_template(
        "Actividad3/ejercicios.html",
        variable=variable.capitalize(),
        accuracy=round(accuracy, 4),
        report=report,
        img_path=img_path
    )



if __name__ == '__main__':
    app.run(debug=True)
