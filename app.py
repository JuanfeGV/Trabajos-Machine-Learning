# app.py
from flask import Flask, render_template, request, jsonify
import time
import json
import os
import Linealregression as Linealregression
from Logicalregression import evaluate, save_confusion_matrix
from GBM import train_and_evaluate  # Import para la actividad 4
from GridWorld import GridWorld
from QLearning import QLearningAgent, train_agent, plot_training_results, simulate_episode, get_training_stats

app = Flask(__name__)

# Directorio para almacenar agentes entrenados
AGENTS_DIR = 'trained_agents'
if not os.path.exists(AGENTS_DIR):
    os.makedirs(AGENTS_DIR)

# Almacenar agentes en memoria durante la sesión
trained_agents = {}

# Cargar agentes guardados previamente
def load_saved_agents():
    """Carga todos los agentes .pkl guardados en trained_agents/"""
    if not os.path.exists(AGENTS_DIR):
        return
    
    for filename in os.listdir(AGENTS_DIR):
        if filename.endswith('_model.pkl'):
            agent_id = filename.replace('_model.pkl', '')
            agent_filepath = os.path.join(AGENTS_DIR, filename)
            
            try:
                agent = QLearningAgent(
                    num_states=25,  # Por defecto 5x5
                    num_actions=4,
                    learning_rate=0.1,
                    discount_factor=0.95,
                    exploration_rate=0.01
                )
                agent.load(agent_filepath)
                
                # Cargar metadatos
                metadata_filepath = os.path.join(AGENTS_DIR, f"{agent_id}_metadata.json")
                metadata = {}
                if os.path.exists(metadata_filepath):
                    with open(metadata_filepath, 'r') as f:
                        metadata = json.load(f)
                
                trained_agents[agent_id] = {
                    'agent': agent,
                    'env': None,  # No se guarda el entorno completo
                    'metadata': metadata
                }
                print(f"[INFO] Agente cargado: {agent_id}")
            except Exception as e:
                print(f"[ERROR] No se pudo cargar {agent_id}: {e}")

load_saved_agents()

# ------------------- Ruta principal -------------------
@app.route('/')
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
def conceptosRL_act2():
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
    variable = request.args.get("var", "antiguedad")  # valor por defecto

    valid_features = {
        "antiguedad": ["Antiguedad"],
        "salario": ["Nivelsalarial(smlv)"],
        "horas_extra": ["Horasextra"],
        "area": ["Areadetrabajo"]
    }

    features = valid_features[variable]

    try:
        model, accuracy, report, conf_matrix = evaluate(features)
        img_path = save_confusion_matrix(conf_matrix, variable)
    except Exception as e:
        import traceback
        return f"<h1>Error al procesar '{variable}'</h1><pre>{traceback.format_exc()}</pre>"

    return render_template(
        "Actividad3/ejercicios.html",
        variable=variable.capitalize(),
        accuracy=round(accuracy, 4),
        report=report,
        img_path=img_path
    )

# ------------------- Actividad 4 (GBM) -------------------
@app.route('/actividad4')
def actividad4():
    return render_template('Actividad4/actividad4.html')

@app.route('/actividad4/conceptosGBM')
def conceptosGBM():
    return render_template('Actividad4/conceptosGBM.html')

@app.route('/actividad4/ejerciciosGBM')
def ejerciciosGBM():
    variable = request.args.get("var", None)

    try:
        accuracy, report, img_path, feature_importances = train_and_evaluate(variable)
    except Exception as e:
        import traceback
        return f"<h1>Error en GBM</h1><pre>{traceback.format_exc()}</pre>"

    return render_template(
        "Actividad4/ejerciciosGBM.html",
        variable=variable.capitalize() if variable else "Todas",
        accuracy=round(accuracy, 4),
        report=report,
        img_path=img_path,
        feature_importances=feature_importances.to_dict(orient="records")
    )

@app.route('/actividad4/ejercicios')
def ejerciciosGBM_old():
    """Mantener compatibilidad con ruta antigua"""
    return ejerciciosGBM()

# ------------------- Actividad 4 (Aprendizaje por Refuerzo) -------------------
@app.route('/actividad4/conceptosRL')
def conceptosRL():
    return render_template('Actividad4/conceptosRL.html')

@app.route('/actividad4/ejerciciosRL')
def ejerciciosRL():
    return render_template('Actividad4/ejerciciosRL.html')

@app.route('/actividad4/agents', methods=['GET'])
def list_saved_agents():
    """Lista todos los agentes guardados con sus metadatos"""
    agents_list = []
    
    if not os.path.exists(AGENTS_DIR):
        return jsonify({'success': True, 'agents': []})
    
    for filename in os.listdir(AGENTS_DIR):
        if filename.endswith('_metadata.json'):
            agent_id = filename.replace('_metadata.json', '')
            metadata_filepath = os.path.join(AGENTS_DIR, filename)
            
            try:
                with open(metadata_filepath, 'r') as f:
                    metadata = json.load(f)
                
                agents_list.append({
                    'agent_id': agent_id,
                    'metadata': metadata,
                    'saved_to': os.path.join(AGENTS_DIR, f"{agent_id}_model.pkl")
                })
            except Exception as e:
                print(f"[ERROR] No se pudo leer {filename}: {e}")
    
    return jsonify({'success': True, 'agents': agents_list})

@app.route('/actividad4/train', methods=['POST'])
def train_rl_agent():
    """Entrena un agente Q-Learning en GridWorld"""
    try:
        data = request.get_json()
        
        num_episodes = int(data.get('num_episodes', 200))
        learning_rate = float(data.get('learning_rate', 0.1))
        discount_factor = float(data.get('discount_factor', 0.95))
        grid_size = int(data.get('grid_size', 5))
        
        # Crear entorno
        env = GridWorld(grid_size=grid_size)
        
        # Crear agente
        agent = QLearningAgent(
            num_states=env.num_states,
            num_actions=env.num_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=1.0
        )
        
        # Entrenar
        rewards, lengths = train_agent(env, agent, num_episodes)
        
        # Generar gráficos
        images = plot_training_results(rewards, lengths)
        
        # Calcular estadísticas
        stats = get_training_stats(rewards)
        
        # Generar visualización del entorno
        env_image = env.render()
        
        # Guardar agente en memoria
        agent_id = f"agent_{len(trained_agents)}"
        trained_agents[agent_id] = {
            'agent': agent,
            'env': env,
            'rewards': rewards,
            'lengths': lengths
        }
        
        # Guardar agente en disco (archivo .pkl)
        agent_filepath = os.path.join(AGENTS_DIR, f"{agent_id}_model.pkl")
        agent.save(agent_filepath)
        
        # Guardar metadatos en JSON
        metadata_filepath = os.path.join(AGENTS_DIR, f"{agent_id}_metadata.json")
        metadata = {
            'grid_size': grid_size,
            'num_episodes': num_episodes,
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'final_reward': float(rewards[-1]) if rewards else 0,
            'avg_reward': float(stats['avg_reward']),
            'max_reward': float(stats['max_reward']),
            'min_reward': float(stats['min_reward'])
        }
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            'success': True,
            'agent_id': agent_id,
            'stats': stats,
            'images': images,
            'env_image': env_image,
            'saved_to': agent_filepath
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

@app.route('/actividad4/simulate', methods=['POST'])
def simulate_rl_agent():
    """Simula un episodio con el agente entrenado"""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id')
        
        if agent_id not in trained_agents:
            return jsonify({
                'success': False,
                'error': 'Agente no encontrado'
            }), 400
        
        agent_data = trained_agents[agent_id]
        agent = agent_data['agent']
        env = agent_data['env']
        
        # Simular episodio
        path, total_reward, success = simulate_episode(env, agent)
        
        # Renderizar trayectoria
        image = env.render(path=path)
        
        return jsonify({
            'success': True,
            'image': image,
            'path': path,
            'total_reward': total_reward,
            'steps': len(path) - 1,
            'success': success
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

# ------------------- Main -------------------
if __name__ == '__main__':
    app.run(debug=True)
