from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/') # Ruta principal
def index():
    return render_template('index.html')

@app.route('/actividad1') # Ruta para la Actividad 1
def actividad1():
    return render_template('actividad1.html')

@app.route('/actividad1/caso1') # Ruta para Actividad1/Caso 1
def caso1():
    return render_template('caso1.html')

@app.route('/actividad1/caso2') # Ruta para Actividad1/Caso 2
def caso2():
    return render_template('caso2.html')

@app.route('/actividad1/caso3') # Ruta para Actividad1/Caso 3
def caso3():
    return render_template('caso3.html')

@app.route('/actividad1/caso4') # Ruta para Actividad1/Caso 4
def caso4():
    return render_template('caso4.html')

@app.route('/actividad2') # Ruta para la Actividad 2
def actividad2():
    return "<h1>Actividad 2 (aún no implementada)</h1>"

@app.route('/actividad3') # Ruta para la Actividad 3
def actividad3():
    return "<h1>Actividad 3 (aún no implementada)</h1>"

if __name__ == '__main__': # Esto sirve para que el código dentro de ese if solo se ejecute cuando el archivo es el programa principal
    app.run(debug=True)