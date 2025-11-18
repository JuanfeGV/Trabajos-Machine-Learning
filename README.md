# Trabajos Machine Learning - Actividades Educativas

Este repositorio contiene una colección de actividades educativas sobre Machine Learning implementadas con Flask. Cada actividad explora diferentes algoritmos y paradigmas de aprendizaje automático.

## Actividades

### Actividad 1: Clustering y Segmentación
- **Contenido:** Técnicas de clustering (K-means, Hierarchical Clustering)
- **Objetivo:** Segmentar datos en grupos sin etiquetar

### Actividad 2: Regresión Lineal
- **Contenido:** Modelado de relaciones lineales entre variables
- **Caso de estudio:** Predicción de glucosa basada en edad y sueño

### Actividad 3: Regresión Logística
- **Contenido:** Clasificación binaria con regresión logística
- **Caso de estudio:** Predicción de comportamiento laboral

### Actividad 4: Aprendizaje por Refuerzo (NUEVO)
- **Contenido:** Q-Learning y entornos tipo GridWorld
- **Objetivo:** Entrenar un agente inteligente que aprende por interacción

## Estructura del Proyecto

```
├── app.py                    # Aplicación principal Flask
├── requirements.txt          # Dependencias Python
├── templates/               # Plantillas HTML
│   ├── base.html           # Template base
│   ├── index.html          # Página principal
│   ├── Actividad1/
│   ├── Actividad2/
│   ├── Actividad3/
│   └── Actividad4/
│       ├── actividad4.html       # Página principal A4
│       ├── conceptosRL.html      # Síntesis teórica con referencias APA
│       └── ejerciciosRL.html     # Caso práctico interactivo
├── static/                  # Archivos estáticos (CSS, imágenes)
├── GridWorld.py            # Entorno de simulación GridWorld
├── QLearning.py            # Implementación del algoritmo Q-Learning
├── Linealregression.py     # Regresión lineal
├── Logicalregression.py    # Regresión logística
└── GBM.py                  # Gradient Boosting Machines
```

## Actividad 4: Aprendizaje por Refuerzo

### 1. Teoría: Conceptos Básicos

La sección **Conceptos Básicos** incluye:

- **Definición General del RL:** Diferencia con aprendizaje supervisado y no supervisado
- **Componentes del Modelo:** Agente, entorno, estados, acciones, recompensas y política
- **Ciclo de Aprendizaje:** Exploración vs. explotación, retorno acumulado, descuento temporal
- **Algoritmos Principales:** 
  - **Q-Learning:** Algoritmo sin modelo, off-policy
  - **SARSA:** Algoritmo on-policy
  - **Deep Q-Network (DQN):** Red neuronal profunda para espacios complejos
- **Buenas Prácticas:** Estabilidad, convergencia, manejo de recompensas

**Referencias (Formato APA 7):**
1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT press.
2. Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmüller, M. (2013). Playing Atari with deep reinforcement learning. *arXiv preprint arXiv:1312.5602*.
3. Thrun, S., & Pratt, L. (Eds.). (1998). *Learning to learn*. Springer Science+Business Media.

### 2. Implementación: GridWorld Environment

#### Características del Entorno

- **Tipo:** Cuadrícula discreta configurable (3x3, 5x5, 7x7)
- **Estados:** Posiciones (x, y) en la cuadrícula
- **Acciones:** 4 movimientos (arriba, abajo, izquierda, derecha)
- **Recompensas:**
  - +1.0 al alcanzar el objetivo
  - -0.1 por cada paso (penalización)
- **Objetivo:** Agente debe navegar desde (0,0) hasta (grid_size-1, grid_size-1)

#### Algoritmo Q-Learning

**Ecuación de Actualización:**
```
Q(s,a) = Q(s,a) + α * [r + γ * max_a Q(s',a) - Q(s,a)]
```

**Parámetros:**
- **α (Learning Rate):** Tasa de aprendizaje [0.01 - 1.0]
- **γ (Discount Factor):** Factor de descuento [0.0 - 1.0]
- **ε (Exploration Rate):** Tasa de exploración inicial [0 - 1]

### 3. Interfaz Web (Flask)

#### Rutas Disponibles

```
GET  /actividad4                    # Página principal
GET  /actividad4/conceptosRL        # Síntesis teórica
GET  /actividad4/ejerciciosRL       # Caso práctico interactivo
POST /actividad4/train              # Entrenar agente
POST /actividad4/simulate           # Simular episodio
```

#### Características de la Interfaz

1. **Panel de Control:**
   - Selector de parámetros (episodios, tasa de aprendizaje, descuento, tamaño de grid)
   - Botones: Entrenar, Simular, Reiniciar

2. **Estadísticas en Tiempo Real:**
   - Total de episodios
   - Recompensa promedio
   - Recompensa máxima
   - Estado del entrenamiento

3. **Visualización:**
   - Gráfico de evolución de recompensas (con promedio móvil)
   - Gráfico de convergencia (pasos por episodio)
   - Visualización del entorno GridWorld
   - Trayectoria del agente durante simulación

#### Flujo de Uso

1. Configurar parámetros (opcional: valores por defecto disponibles)
2. Click en "Iniciar Entrenamiento"
3. Esperar a que se entrene (progreso mostrado)
4. Ver gráficos y estadísticas
5. Click en "Simular Episodio" para ver al agente en acción
6. Experimentar con diferentes parámetros

## Instalación y Uso

### Requisitos Previos
- Python 3.8+
- pip

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/JuanfeGV/Trabajos-Machine-Learning.git
cd Trabajos-Machine-Learning
```

2. Crear entorno virtual (recomendado):
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# o
source venv/bin/activate      # macOS/Linux
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

### Ejecutar la Aplicación

```bash
python app.py
```

Luego acceder a `http://localhost:5000` en el navegador.

## Resultados Esperados

### Entrenamiento Exitoso

- **Recompensa Promedio:** Aumenta progresivamente
- **Convergencia:** Número de pasos disminuye (agente aprende rutas más cortas)
- **Tasa de Éxito:** 90%+ después de 200 episodios (grid 5x5)

### Interpretación de Parámetros

| Parámetro | Valor Bajo | Valor Alto | Efecto |
|-----------|-----------|-----------|--------|
| Episodios | Entrenamiento rápido | Convergencia garantizada | Más episodios = mejor |
| α | Convergencia lenta | Inestabilidad | Balance en 0.1 |
| γ | Miope, corto plazo | Visión a futuro | 0.95 es típico |

## Flujo de Trabajo Git

Este proyecto sigue la rama `A12_ReinforcementLearning`:

```bash
# Crear/cambiar a rama
git checkout -b A12_ReinforcementLearning

# Commits atómicos realizados:
# - feat: Agregar GridWorld.py (entorno de RL)
# - feat: Implementar QLearning.py (algoritmo Q-Learning)
# - feat: Agregar rutas Flask para Actividad 4
# - feat: Crear interfaz web para Actividad 4
# - docs: Actualizar README con documentación RL

# Push y Pull Request
git push origin A12_ReinforcementLearning
```

## Mejoras Futuras

- [ ] Implementación de SARSA
- [ ] Deep Q-Learning con redes neuronales
- [ ] Policy Gradient methods
- [ ] Entornos más complejos (laberintos, obstáculos)
- [ ] Visualización en 3D
- [ ] Guardado/carga de modelos entrenados
- [ ] Comparación de algoritmos

## Criterios de Evaluación (5 puntos)

- **Conceptos Básicos (1 pt):** Síntesis teórica clara con referencias APA
- **Implementación (1 pt):** GridWorld + Q-Learning funcionando correctamente
- **Entrenamiento (1 pt):** Convergencia evidenciada en gráficos
- **Interfaz Flask (1 pt):** Interfaz web operativa e intuitiva
- **GitHub (1 pt):** Rama, commits descriptivos, documentación

## Autor

**Equipo de Desarrollo:**
- Carrera: Ingeniería de Software
- Universidad: [Nombre de la Universidad]
- Periodo: 2025-1

## Licencia

Este proyecto es de uso educativo y está disponible bajo licencia MIT.

---

**Última actualización:** Noviembre 17, 2025

