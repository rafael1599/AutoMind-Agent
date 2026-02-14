
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from torch.distributions.categorical import Categorical

# Asegurar que el path incluya scripts para importar taxi_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from taxi_env import TaxiEnv

# --- CONFIGURACION ---
MODEL_PATH = "models/automind_final.pth"
NUM_EPISODES = 5
RENDER_DELAY = 0.2
GRID_SIZE = 5

class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = np.array(observation_space.shape).prod()
        act_dim = action_space.n
        
        # --- ACTOR BACKBONE ---
        self.actor_features = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU()
        )
        self.actor_lstm = nn.LSTM(128, 128)
        self.actor_ln = nn.LayerNorm(128)
        self.actor_head = nn.Linear(128, act_dim)
        
        # --- CRITIC BACKBONE ---
        self.critic_features = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU()
        )
        self.critic_lstm = nn.LSTM(128, 128)
        self.critic_ln = nn.LayerNorm(128)
        self.critic_head = nn.Linear(128, 1)

    def get_action_and_value(self, x, actor_state, critic_state, done, action=None):
        a_hidden = self.actor_features(x)
        # Batch size handling for single inference
        if len(x.shape) == 1:
             x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        a_hidden = a_hidden.reshape((-1, batch_size, 128))
        # Dummy done for inference if not batched correctly or just ignore LSTM reset logic during simple inference if done is bool
        # For simplicity in demo, we assume done is handled outside or passed as tensor
        
        # LSTM forward
        h_a, actor_state = self.actor_lstm(a_hidden, actor_state)
        h_a = self.actor_ln(h_a.squeeze(0))
        logits = self.actor_head(h_a)
        probs = Categorical(logits=logits)
        
        # Critic forward (optional for demo, but good for debug)
        c_hidden = self.critic_features(x).reshape((-1, batch_size, 128))
        h_c, critic_state = self.critic_lstm(c_hidden, critic_state)
        h_c = self.critic_ln(h_c.squeeze(0))
        value = self.critic_head(h_c)
        
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, actor_state, critic_state

def run_demo():
    print("\n" + "="*50)
    print("  AUTOMIND ULTRA: VISUALIZATION (LiDAR + Dual Brain)")
    print("="*50 + "\n")

    # Iniciar entorno real Nivel 3 (Muros complejos)
    env = TaxiEnv(grid_size=GRID_SIZE)
    env.set_level(3) # Test en el nivel difícil
    # Forzar obstáculos de nivel 3 si no se generan por defecto en reset
    # env.generate_random_obstacles(6) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env.observation_space, env.action_space).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            agent.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            print(f"[OK] Modelo cargado: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Fallo al cargar {MODEL_PATH}: {e}")
            return
    else:
        print(f"[ERROR] No se encontro {MODEL_PATH}. Entrena primero!")
        return

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        step = 0
        
        # Initialize separate LSTM states
        a_state = (torch.zeros(1, 1, 128).to(device), torch.zeros(1, 1, 128).to(device))
        c_state = (torch.zeros(1, 1, 128).to(device), torch.zeros(1, 1, 128).to(device))
        
        total_reward = 0
        print(f"\n--- EPISODIO {ep+1} (Nivel {env.current_level}) ---")
        
        while not done and step < 50:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            done_t = torch.tensor([0], dtype=torch.float32).to(device) # Dummy done
            
            with torch.no_grad():
                action, _, _, val, a_state, c_state = agent.get_action_and_value(obs_t, a_state, c_state, done_t)
            
            action_idx = action.item()
            obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
            total_reward += reward
            
            # Obtener datos de LiDAR para visualización
            lidar = info.get('lidar', [0,0,0,0])
            
            # Visualización
            render_grid(env, step, action_idx, val.item(), reward, lidar)
            time.sleep(RENDER_DELAY)
            step += 1
            
        status = "EXITO" if info.get('is_success') else "FALLO"
        print(f"\nResultado: {status} en {step} pasos. Recompensa Total: {total_reward:.2f}")
        time.sleep(1)

def render_grid(env, step, action, val, rew, lidar):
    actions = ["Sur", "Norte", "Este", "Oeste", "Pickup", "Dropoff"]
    
    grid = [[" " for _ in range(5)] for _ in range(5)]
    for r, c in env.obstacles: grid[r][c] = "X"
    loc_names = ["R", "G", "Y", "B"]
    for i, (r, c) in enumerate(env.locs): grid[r][c] = loc_names[i]
    
    if not env.has_passenger:
        pr, pc = env.pass_row, env.pass_col
        grid[pr][pc] = "P"
    
    tr, tc = env.taxi_row, env.taxi_col
    grid[tr][tc] = "T" # Taxi con pasajero o sin él

    lidar_str = f"N:{int(lidar[0])} S:{int(lidar[1])} E:{int(lidar[2])} W:{int(lidar[3])}"
    print(f"\rStep:{step:2d} | Act:{actions[action]:7s} | Val:{val:5.2f} | Rew:{rew:3.0f} | LiDAR:[{lidar_str}]")
    print("  +---+---+---+---+---+")
    for row_idx, row in enumerate(grid):
        line = "  | "
        for cell in row:
            if cell == "X": line += "XXX"
            elif cell == "T": line += "TAX"
            elif cell == "P": line += "PAS"
            else: line += f" {cell} "
            line += " |"
        print(line)
    print("  +---+---+---+---+---+")

if __name__ == "__main__":
    run_demo()
