
import asyncio
import json
import websockets
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
NUM_EPISODES = 1000  # Casi indefinido para demo continua
RENDER_DELAY = 0.8   # Aumentado para dar tiempo a la interpolación suave en Unreal 5.6
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

        # --- MEMORIA A LARGO PLAZO (TARGET CRITIC) ---
        self.target_critic_features = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU()
        )
        self.target_critic_lstm = nn.LSTM(128, 128)
        self.target_critic_ln = nn.LayerNorm(128)
        self.target_critic_head = nn.Linear(128, 1)

    def get_action_and_value(self, x, actor_state, critic_state, done, action=None):
        a_hidden = self.actor_features(x)
        if len(x.shape) == 1: x = x.unsqueeze(0)
        batch_size = x.shape[0]
        a_hidden = a_hidden.reshape((-1, batch_size, 128))
        h_a, actor_state = self.actor_lstm(a_hidden, actor_state)
        h_a = self.actor_ln(h_a.squeeze(0))
        logits = self.actor_head(h_a)
        probs = Categorical(logits=logits)
        
        c_hidden = self.critic_features(x).reshape((-1, batch_size, 128))
        h_c, critic_state = self.critic_lstm(c_hidden, critic_state)
        h_c = self.critic_ln(h_c.squeeze(0))
        value = self.critic_head(h_c)
        
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, actor_state, critic_state

async def send_telemetry(websocket, payload):
    try:
        await websocket.send(json.dumps(payload))
    except (websockets.ConnectionClosed, websockets.ConnectionClosedOK):
        print(f"[ERROR] Conexión cerrada.")
        return False
    return True

async def run_telemetry_session(websocket):
    print(f"\n[SERVER] Cliente conectado: {websocket.remote_address}")
    
    env = TaxiEnv(grid_size=GRID_SIZE)
    env.set_level(3)
    
    # Reintentar generación hasta éxito
    max_retries = 100
    success = False
    for i in range(max_retries):
        if env.generate_random_obstacles(6):
            success = True
            break
        env.reset()
    
    if success:
        print(f"[OK] Obstáculos generados exitosamente: {len(env.obstacles)}")
        print(f"     Coordenadas: {list(env.obstacles)}")
    else:
        print("[WARNING] No se pudieron generar obstáculos navegables tras 100 intentos.")
    
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
        print(f"[ERROR] No se encontro {MODEL_PATH}.")
        return

    # Enviar estado inicial inmediatamente al conectar para sincronizar visualizador
    init_sync_payload = {
        "type": "init",
        "episode": 0,
        "grid_size": GRID_SIZE,
        "obstacles": [{"x": int(r), "y": int(c)} for r, c in env.obstacles],
        "locations": [{"name": n, "x": int(r), "y": int(c)} for n, (r, c) in zip(["R", "G", "Y", "B"], env.locs)]
    }
    await send_telemetry(websocket, init_sync_payload)
    print(f"[SYNC] Enviada configuración inicial al cliente ({len(env.obstacles)} obstáculos)")

    telemetry_history = []
    
    try:
        # Definir nivel inicial
        current_level = 1
        obstacles_per_level = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}

        for ep in range(NUM_EPISODES):
            # Progresión de nivel cada 5 episodios
            if ep > 0 and ep % 5 == 0 and current_level < 5:
                current_level += 1
                print(f"\n" + "!"*20 + f" SUBIDA DE NIVEL: {current_level} " + "!"*20)
            
            env.set_level(current_level)
            num_obs = obstacles_per_level.get(current_level, 6)
            
            # Regenerar obstáculos dinámicamente para este episodio
            success = False
            for _ in range(50):
                if env.generate_random_obstacles(num_obs):
                    success = True
                    break
                env.reset()
            
            obs, _ = env.reset()
            done = False
            step = 0
            
            # Reset LSTM states
            a_state = (torch.zeros(1, 1, 128).to(device), torch.zeros(1, 1, 128).to(device))
            c_state = (torch.zeros(1, 1, 128).to(device), torch.zeros(1, 1, 128).to(device))
            
            print(f"\n--- EPISODIO {ep+1} (Nivel {current_level} - {num_obs} Obstáculos) ---")
            
            # Sincronizar configuración con Unreal
            initial_payload = {
                "type": "init",
                "episode": ep + 1,
                "level": current_level,
                "grid_size": GRID_SIZE,
                "obstacles": [{"x": int(r), "y": int(c)} for r, c in env.obstacles],
                "locations": [{"name": n, "x": int(r), "y": int(c)} for n, (r, c) in zip(["R", "G", "Y", "B"], env.locs)]
            }
            if not await send_telemetry(websocket, initial_payload): break
            
            await asyncio.sleep(2.0) # Pause so Unreal shows "START!" message
            
            # Initial Orientation (Default North)
            orientation = "North" 
            
            while not done and step < 50:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                done_t = torch.tensor([0], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    action, _, entropy, val, a_state, c_state = agent.get_action_and_value(obs_t, a_state, c_state, done_t)
                
                action_idx = action.item()
                
                # Update Orientation based on action
                # 0: Sur, 1: Norte, 2: Este, 3: Oeste
                if action_idx == 0: orientation = "South"
                elif action_idx == 1: orientation = "North"
                elif action_idx == 2: orientation = "East"
                elif action_idx == 3: orientation = "West"
                
                obs, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated
                
                lidar = info.get('lidar', [0,0,0,0]).tolist() if isinstance(info.get('lidar'), np.ndarray) else [0,0,0,0]
                
                payload = {
                    "type": "step",
                    "step": step,
                    "position": {
                        "x": int(env.taxi_row), 
                        "y": int(env.taxi_col), 
                        "orientation": orientation
                    },
                    "sensors": {
                        "N": int(lidar[0]), 
                        "S": int(lidar[1]), 
                        "E": int(lidar[2]), 
                        "W": int(lidar[3])
                    },
                    "brain": {
                        "value": float(val.item()), 
                        "entropy": float(entropy.item()),
                        "action": action_idx,
                        "action_name": ["South", "North", "East", "West", "Pickup", "Dropoff"][action_idx]
                    },
                    "status": {
                        "has_passenger": bool(env.has_passenger), 
                        "passenger_pos": {"x": int(env.pass_row), "y": int(env.pass_col)},
                        "dest_pos": {"x": int(env.dest_row), "y": int(env.dest_col)},
                        "is_success": bool(info.get('is_success', False)), 
                        "reward": float(reward)
                    }
                }
                
                telemetry_history.append(payload)
                
                if not await send_telemetry(websocket, payload): break
                
                # Console Feedback
                print(f"\rStep:{step:2d} | Act:{payload['brain']['action_name']:7s} | Val:{payload['brain']['value']:5.2f} | Rew:{reward:3.0f}", end="")
                
                await asyncio.sleep(RENDER_DELAY)
                step += 1
            
            # Send End of Episode
            end_payload = {"type": "episode_end", "success": bool(info.get('is_success', False)), "total_steps": step}
            if not await send_telemetry(websocket, end_payload): break
            await asyncio.sleep(1.0) # Pause between episodes

    except Exception as e:
        print(f"\n[ERROR] Excepción en sesión: {e}")
    finally:
        # Save History
        with open("telemetry_history.json", "w") as f:
            json.dump(telemetry_history, f, indent=2)
        print("\n\n[INFO] Historial guardado en telemetry_history.json")

async def main():
    print("\n" + "="*50)
    print("  AUTOMIND 3D TELEMETRY SERVER")
    print("  Listening on: ws://localhost:8765")
    print("="*50 + "\n")
    
    try:
        async with websockets.serve(run_telemetry_session, "localhost", 8765):
            await asyncio.get_running_loop().create_future()  # Run forever
    except OSError as e:
        if e.errno == 10048:
            print("\n" + "!"*50)
            print(" [ERROR] El puerto 8765 ya está en uso.")
            print(" Por favor, cierra otras instancias de Python o")
            print(" usa el comando: taskkill /F /IM python.exe")
            print("!"*50 + "\n")
        else:
            raise e

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SERVER] Detenido por usuario.")
