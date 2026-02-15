
import sys
import os
import msvcrt # Librería estándar de Windows para capturar teclas
import time

# Asegurar que el path incluya scripts para importar taxi_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from taxi_env import TaxiEnv

def get_key():
    """Captura teclas en Windows (incluyendo flechas)."""
    ch = msvcrt.getch()
    if ch == b'\xe0': # Tecla especial (flechas)
        ch = msvcrt.getch()
        return {b'H': 'up', b'P': 'down', b'K': 'left', b'M': 'right'}.get(ch, None)
    return ch.decode('utf-8').lower()

def render_manual(env, reward, total_reward, msg=""):
    os.system('cls' if os.name == 'nt' else 'clear')
    actions_map = ["Sur", "Norte", "Este", "Oeste", "Pickup", "Dropoff"]
    
    print("="*40)
    print("  AUTO-DRIVE MANUAL TEST (HUMAN-IN-THE-LOOP)")
    print("="*40)
    print(f" Flechas: Mover | P: Pickup | D: Dropoff | Q: Salir")
    print(f" Recompensa: {reward:+.1f} | Total: {total_reward:+.1f}")
    if msg: print(f" LOG: {msg}")
    print("-" * 40)
    
    # Grid construction
    grid = [[" " for _ in range(5)] for _ in range(5)]
    for r, c in env.obstacles: grid[r][c] = "X"
    loc_names = ["R", "G", "Y", "B"]
    for i, (r, c) in enumerate(env.locs): 
        grid[env.locs[i][0]][env.locs[i][1]] = loc_names[i]
    
    if not env.has_passenger:
        pr, pc = env.locs[env.pass_idx]
        grid[pr][pc] = "P" # P sobreescribe estación
    
    grid[env.taxi_row][env.taxi_col] = "T"
    
    print("  +---+---+---+---+---+")
    for row in grid:
        line = "  | " + " | ".join([c if c != "X" else "XXX" for c in row]) + " |"
        print(line)
    print("  +---+---+---+---+---+")
    
    # Info extra
    dest = ["R", "G", "Y", "B"][env.dest_idx]
    status = "RECOGER PASAJERO" if not env.has_passenger else f"LLEVAR A {dest}"
    print(f" OBJETIVO: {status}")

def run_manual():
    # Usar los obstáculos de L4 que el agente no pudo superar
    obstacles = [(0,1), (1,1), (2,1), (4,2), (0,3), (1,3), (2,3)]
    env = TaxiEnv(grid_size=5, obstacles=obstacles)
    obs, _ = env.reset()
    
    total_reward = 0
    last_reward = 0
    msg = "Motor encendido. Suerte, conductor."
    
    while True:
        render_manual(env, last_reward, total_reward, msg)
        key = get_key()
        
        action = None
        if key == 'up': action = 1
        elif key == 'down': action = 0
        elif key == 'left': action = 3
        elif key == 'right': action = 2
        elif key == 'p': action = 4
        elif key == 'd': action = 5
        elif key == 'q': break
        
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            last_reward = reward
            total_reward += reward
            
            # Feedback de colisión o éxito
            if reward <= -30: msg = "¡CRASH! -30 puntos."
            elif reward == 10: msg = "¡Pasajero a bordo! (+10)"
            elif reward == 50: 
                render_manual(env, reward, total_reward, "¡MISIÓN CUMPLIDA! (+50)")
                print("\n\nPresiona cualquier tecla para nuevo viaje...")
                msvcrt.getch()
                env.reset()
                total_reward = 0
                msg = "Reiniciando..."
            elif reward == -10: msg = "Acción ilegal. (-10)"
            else: msg = ""
            
            if terminated or truncated:
                env.reset()

if __name__ == "__main__":
    run_manual()
