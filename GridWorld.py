"""
GridWorld - Entorno simple para Aprendizaje por Refuerzo
Este módulo implementa un entorno tipo GridWorld donde un agente
debe aprender a navegar desde una posición inicial a una posición objetivo.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import io
import base64
from PIL import Image


class GridWorld:
    """
    Entorno GridWorld: una cuadrícula donde el agente debe llegar al objetivo.
    
    Estados: Posiciones (x, y) en la cuadrícula
    Acciones: Arriba (0), Abajo (1), Izquierda (2), Derecha (3)
    Recompensas: +1 al llegar al objetivo, -0.1 por paso
    """
    
    def __init__(self, grid_size=5, agent_start=(0, 0), goal=(4, 4)):
        """
        Inicializa el entorno GridWorld.
        
        Args:
            grid_size: Tamaño de la cuadrícula (grid_size x grid_size)
            agent_start: Posición inicial del agente (x, y)
            goal: Posición objetivo (x, y)
        """
        self.grid_size = grid_size
        self.agent_start = agent_start
        self.goal = goal
        self.agent_pos = list(agent_start)
        
        # Acciones: 0=arriba, 1=abajo, 2=izquierda, 3=derecha
        self.actions = ['up', 'down', 'left', 'right']
        self.action_deltas = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        
        self.num_states = grid_size * grid_size
        self.num_actions = 4
        self.steps = 0
        self.max_steps = grid_size * grid_size * 2  # Límite de pasos por episodio
        
    def reset(self):
        """Reinicia el entorno a la posición inicial."""
        self.agent_pos = list(self.agent_start)
        self.steps = 0
        return self._get_state()
    
    def _get_state(self):
        """Convierte la posición (x, y) a un índice de estado único."""
        x, y = self.agent_pos
        return y * self.grid_size + x
    
    def _pos_from_state(self, state):
        """Convierte un índice de estado a coordenadas (x, y)."""
        y = state // self.grid_size
        x = state % self.grid_size
        return (x, y)
    
    def step(self, action):
        """
        Ejecuta una acción y retorna (nuevo_estado, recompensa, terminado, info).
        
        Args:
            action: Índice de acción (0-3)
            
        Returns:
            state: Nuevo estado
            reward: Recompensa obtenida
            done: Si el episodio terminó
            info: Información adicional
        """
        action_name = self.actions[action]
        dx, dy = self.action_deltas[action_name]
        
        # Calcular nueva posición
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        
        # Verificar límites
        new_x = max(0, min(new_x, self.grid_size - 1))
        new_y = max(0, min(new_y, self.grid_size - 1))
        
        self.agent_pos = [new_x, new_y]
        self.steps += 1
        
        # Calcular recompensa
        reward = -0.1  # Penalización por cada paso
        done = False
        
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0  # Recompensa por llegar al objetivo
            done = True
        elif self.steps >= self.max_steps:
            done = True
        
        state = self._get_state()
        
        return state, reward, done, {"pos": tuple(self.agent_pos), "steps": self.steps}
    
    def render(self, agent_pos=None, path=None):
        """
        Visualiza el estado actual del entorno.
        
        Args:
            agent_pos: Posición del agente a visualizar (por defecto la actual)
            path: Lista de posiciones para mostrar trayectoria
            
        Returns:
            Buffer de imagen en base64
        """
        if agent_pos is None:
            agent_pos = tuple(self.agent_pos)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Dibujar cuadrícula
        for i in range(self.grid_size + 1):
            ax.axhline(i, color='gray', linewidth=0.5)
            ax.axvline(i, color='gray', linewidth=0.5)
        
        # Dibujar objetivo
        goal_rect = patches.Rectangle(
            (self.goal[0], self.goal[1]), 1, 1,
            linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.7
        )
        ax.add_patch(goal_rect)
        ax.text(self.goal[0] + 0.5, self.goal[1] + 0.5, 'G',
                ha='center', va='center', fontsize=20)
        
        # Dibujar trayectoria si existe
        if path:
            path_x = [p[0] + 0.5 for p in path[:-1]]
            path_y = [p[1] + 0.5 for p in path[:-1]]
            ax.plot(path_x, path_y, 'b--', alpha=0.5, linewidth=1)
        
        # Dibujar agente
        agent_rect = patches.Rectangle(
            (agent_pos[0], agent_pos[1]), 1, 1,
            linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.8
        )
        ax.add_patch(agent_rect)
        ax.text(agent_pos[0] + 0.5, agent_pos[1] + 0.5, 'A',
                ha='center', va='center', fontsize=20)
        
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(self.grid_size + 1))
        ax.set_yticks(range(self.grid_size + 1))
        ax.set_title('GridWorld Environment', fontsize=14, fontweight='bold')
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
