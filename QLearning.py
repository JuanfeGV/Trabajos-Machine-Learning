"""
QLearning - Algoritmo de Aprendizaje por Refuerzo
Implementación de Q-Learning para entrenar un agente en GridWorld.
"""

import numpy as np
import pickle
import json
import os
from GridWorld import GridWorld
import matplotlib.pyplot as plt
import base64
import io


class QLearningAgent:
    """
    Agente que aprende usando Q-Learning.
    
    Q-Learning es un algoritmo de aprendizaje por refuerzo sin modelo que
    aprende el valor óptimo de acciones en cada estado.
    """
    
    def __init__(self, num_states, num_actions, learning_rate=0.1, 
                 discount_factor=0.95, exploration_rate=1.0):
        """
        Inicializa el agente Q-Learning.
        
        Args:
            num_states: Número total de estados
            num_actions: Número total de acciones
            learning_rate (α): Tasa de aprendizaje
            discount_factor (γ): Factor de descuento para recompensas futuras
            exploration_rate (ε): Tasa de exploración inicial
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995  # Decay para reducir exploración
        self.min_exploration_rate = 0.01
        
        # Tabla Q inicializada en ceros
        self.q_table = np.zeros((num_states, num_actions))
    
    def choose_action(self, state, training=True):
        """
        Selecciona una acción usando ε-greedy.
        
        Args:
            state: Estado actual
            training: Si es True, usa exploración. Si es False, es pura explotación.
            
        Returns:
            Índice de acción seleccionada
        """
        if training and np.random.random() < self.exploration_rate:
            # Exploración: acción aleatoria
            return np.random.randint(0, self.num_actions)
        else:
            # Explotación: acción con mayor valor Q
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Actualiza la tabla Q usando la ecuación de Q-Learning.
        
        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        
        Args:
            state: Estado anterior
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Nuevo estado
            done: Si el episodio terminó
        """
        # Valor máximo Q del siguiente estado
        max_next_q = np.max(self.q_table[next_state]) if not done else 0
        
        # Ecuación de Q-Learning
        current_q = self.q_table[state, action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state, action] = new_q
    
    def decay_exploration(self):
        """Reduce la tasa de exploración después de cada episodio."""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def get_policy(self):
        """
        Obtiene la política greedy actual.
        
        Returns:
            Array de acciones óptimas para cada estado
        """
        return np.argmax(self.q_table, axis=1)
    
    def save(self, filepath):
        """Guarda el modelo Q-Learning."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filepath):
        """Carga un modelo Q-Learning entrenado."""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)


def train_agent(env, agent, num_episodes=200):
    """
    Entrena el agente en el entorno.
    
    Args:
        env: Entorno GridWorld
        agent: Agente QLearningAgent
        num_episodes: Número de episodios de entrenamiento
        
    Returns:
        rewards: Lista de recompensas acumuladas por episodio
        lengths: Lista de longitud de episodios
    """
    rewards = []
    lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Seleccionar y ejecutar acción
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Actualizar Q-table
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Reducir exploración
        agent.decay_exploration()
        
        rewards.append(total_reward)
        lengths.append(steps)
    
    return rewards, lengths


def plot_training_results(rewards, lengths):
    """
    Crea gráficos del entrenamiento.
    
    Args:
        rewards: Lista de recompensas por episodio
        lengths: Lista de longitudes de episodios
        
    Returns:
        Diccionario con imágenes en base64
    """
    images = {}
    
    # Gráfico 1: Recompensa promedio móvil
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 10
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(len(rewards)), rewards, alpha=0.3, label='Recompensa por episodio')
    ax.plot(range(window-1, len(rewards)), moving_avg, label=f'Promedio móvil (ventana={window})', linewidth=2)
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Recompensa acumulada')
    ax.set_title('Evolución de Recompensas durante el Entrenamiento')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    images['rewards'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
    plt.close()
    
    # Gráfico 2: Longitud de episodios
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lengths, alpha=0.6, color='orange')
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Pasos en el episodio')
    ax.set_title('Longitud de Episodios (Convergencia)')
    ax.grid(True, alpha=0.3)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    images['lengths'] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"
    plt.close()
    
    return images


def simulate_episode(env, agent, max_steps=None):
    """
    Simula un episodio completo con la política aprendida.
    
    Args:
        env: Entorno GridWorld
        agent: Agente entrenado
        max_steps: Número máximo de pasos (opcional)
        
    Returns:
        path: Lista de posiciones visitadas
        total_reward: Recompensa acumulada
        success: Si alcanzó el objetivo
    """
    state = env.reset()
    path = [tuple(env.agent_pos)]
    total_reward = 0
    done = False
    steps = 0
    max_steps = max_steps or env.max_steps
    
    while not done and steps < max_steps:
        # Usar política greedy pura (sin exploración)
        action = agent.choose_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        
        path.append(tuple(env.agent_pos))
        total_reward += reward
        state = next_state
        steps += 1
    
    success = tuple(env.agent_pos) == env.goal
    
    return path, total_reward, success


def get_training_stats(rewards):
    """Calcula estadísticas de entrenamiento."""
    if not rewards:
        return {
            'total_episodes': 0,
            'avg_reward': 0.0,
            'max_reward': 0.0,
            'min_reward': 0.0,
            'final_avg': 0.0,
            'convergence': 0.0
        }
    
    final_window = min(10, len(rewards))
    initial_window = min(10, len(rewards))
    
    return {
        'total_episodes': len(rewards),
        'avg_reward': float(np.mean(rewards)),
        'max_reward': float(np.max(rewards)),
        'min_reward': float(np.min(rewards)),
        'final_avg': float(np.mean(rewards[-final_window:])),
        'convergence': float(np.mean(rewards[-final_window:]) - np.mean(rewards[:initial_window]))
    }
