
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TaxiEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, grid_size=5, obstacles=None):
        super(TaxiEnv, self).__init__()
        self.grid_size = grid_size
        self.window_size = 5
        self.obstacles = set(obstacles) if obstacles else set()
        self.action_space = spaces.Discrete(6)
        # 25 (grid) + 9 (telemetry) + 4 (LiDAR sensors) = 38
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(38,), dtype=np.float32
        )
        
        # Landmarks fijos
        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)] 
        self.current_level = 0
        self.collision_penalty = -5 # Inicia suave
        self.max_steps = 50
        self.current_step = 0
        
        # Tracking
        self.visit_counts = {}
        self.visited_zones = set()
        self.transition_history = []
        
        self.reset()

    def set_level(self, level):
        self.current_level = level
        self.collision_penalty = -5 if level <= 1 else -25

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # --- LÓGICA DE SPAWN POR PROXIMIDAD (Nivel 0) ---
        if self.current_level == 0:
            # Taxi cerca del pasajero
            self.pass_idx = self.np_random.integers(0, 4)
            self.pass_row, self.pass_col = self.locs[self.pass_idx]
            
            # Taxi a max 2 celdas de distancia
            while True:
                self.taxi_row = np.clip(self.pass_row + self.np_random.integers(-2, 3), 0, 4)
                self.taxi_col = np.clip(self.pass_col + self.np_random.integers(-2, 3), 0, 4)
                if (self.taxi_row, self.taxi_col) not in self.obstacles: break
            
            # Destino cerca del pasajero
            while True:
                self.dest_idx = self.np_random.integers(0, 4)
                if self.dest_idx != self.pass_idx: break
            self.dest_row, self.dest_col = self.locs[self.dest_idx]
        else:
            # Spawn aleatorio total (Nivel 1+)
            while True:
                self.taxi_row = self.np_random.integers(0, self.grid_size)
                self.taxi_col = self.np_random.integers(0, self.grid_size)
                if (self.taxi_row, self.taxi_col) not in self.obstacles: break
                
            self.pass_idx = self.np_random.integers(0, 4)
            self.pass_row, self.pass_col = self.locs[self.pass_idx]
            while True:
                self.dest_idx = self.np_random.integers(0, 4)
                if self.dest_idx != self.pass_idx: break
            self.dest_row, self.dest_col = self.locs[self.dest_idx]

        self.has_passenger = 0
        self.visit_counts = {}
        self.visited_zones = set()
        self.transition_history = []
        return self._get_obs(), {}

    def step(self, action):
        prev_row, prev_col = self.taxi_row, self.taxi_col
        prev_potential = self._calculate_potential()
        
        reward = -1 # Step Penalty constante
        terminated = False
        truncated = False
        dropoff_success = False

        # --- SENSORES DE PROXIMIDAD (Nueva Lógica) ---
        sensors = self._get_proximity_sensors()
        if sensors.any():
            reward -= 1.0 # Penalización por aviso de proximidad

        # Movimiento
        new_row, new_col = self.taxi_row, self.taxi_col
        if action == 0: new_row = min(self.taxi_row + 1, 4)
        elif action == 1: new_row = max(self.taxi_row - 1, 0)
        elif action == 2: new_col = min(self.taxi_col + 1, 4)
        elif action == 3: new_col = max(self.taxi_col - 1, 0)
        elif action == 4: # Pickup
            if (self.taxi_row == self.pass_row and self.taxi_col == self.pass_col and self.has_passenger == 0):
                self.has_passenger = 1
                reward = 10
            else: reward = -2 
        elif action == 5: # Dropoff
            if (self.taxi_row == self.dest_row and self.taxi_col == self.dest_col and self.has_passenger == 1):
                self.has_passenger = 0
                terminated = True
                reward = 50
                dropoff_success = True
            else: reward = -2

        # Colisiones y Bordes (Justo pero Firme: -25)
        if action < 4:
            hit_obstacle = (new_row, new_col) in self.obstacles
            hit_border = (new_row == prev_row and new_col == prev_col) and not hit_obstacle
            
            if hit_obstacle or hit_border:
                reward = -25 # Estándar de seguridad firme
            else:
                self.taxi_row, self.taxi_col = new_row, new_col

        # Reward Shaping
        current_potential = self._calculate_potential()
        if (action == 4 and reward == 10) or (action == 5 and reward == 50):
            shaping = 0 
        else:
            shaping = 0.99 * current_potential - prev_potential
        
        if not terminated: reward += shaping

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        info = {
            "is_success": dropoff_success,
            "step_count": self.current_step,
            "lidar": sensors # Telemetría de depuración
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _get_proximity_sensors(self):
        """Virtual LiDAR: 1.0 si hay muro adyacente en [N, S, E, O]."""
        # N: r-1, S: r+1, E: c+1, W: c-1
        check_pos = [
            (self.taxi_row - 1, self.taxi_col), # N
            (self.taxi_row + 1, self.taxi_col), # S
            (self.taxi_row, self.taxi_col + 1), # E
            (self.taxi_row, self.taxi_col - 1)  # W
        ]
        sensors = np.zeros(4, dtype=np.float32)
        for i, (r, c) in enumerate(check_pos):
            if r < 0 or r > 4 or c < 0 or c > 4 or (r, c) in self.obstacles:
                sensors[i] = 1.0
        return sensors

    def _calculate_potential(self):
        target = (self.dest_row, self.dest_col) if self.has_passenger else (self.pass_row, self.pass_col)
        dist = abs(self.taxi_row - target[0]) + abs(self.taxi_col - target[1])
        return 10.0 - dist

    def _get_obs(self):
        # Brújula absoluta
        telemetry = np.array([
            self.taxi_row/4, self.taxi_col/4, 
            self.pass_row/4, self.pass_col/4,
            self.dest_row/4, self.dest_col/4,
            self.pass_idx/3, self.dest_idx/3, 
            self.has_passenger
        ], dtype=np.float32)
        
        # Inyectar LiDAR a la telemetría
        sensors = self._get_proximity_sensors()
        telemetry = np.concatenate((telemetry, sensors))
        
        grid = np.zeros(25, dtype=np.float32)
        if not self.has_passenger:
            p_r, p_c = self.pass_row - self.taxi_row + 2, self.pass_col - self.taxi_col + 2
            if 0 <= p_r < 5 and 0 <= p_c < 5: grid[p_r*5 + p_c] = 0.5
        d_r, d_c = self.dest_row - self.taxi_row + 2, self.dest_col - self.taxi_col + 2
        if 0 <= d_r < 5 and 0 <= d_c < 5: grid[d_r*5 + d_c] = 1.0
        return np.concatenate((grid, telemetry))

    def generate_random_obstacles(self, count):
        """BFS para asegurar navegabilidad."""
        for _ in range(50):
            new_obs = set()
            while len(new_obs) < count:
                r, c = np.random.randint(0, 5, 2)
                if (r, c) not in self.locs and (r, c) != (self.taxi_row, self.taxi_col):
                    new_obs.add((r, c))
            if self._is_reachable(new_obs):
                self.obstacles = new_obs
                return True
        return False

    def _is_reachable(self, obs):
        def can(s, g, o):
            q, v = [s], {s}
            while q:
                curr = q.pop(0)
                if curr == g: return True
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr, nc = curr[0]+dr, curr[1]+dc
                    if 0<=nr<5 and 0<=nc<5 and (nr,nc) not in o and (nr,nc) not in v:
                        v.add((nr,nc)); q.append((nr,nc))
            return False
        tp, pp, dp = (self.taxi_row, self.taxi_col), (self.pass_row, self.pass_col), (self.dest_row, self.dest_col)
        return can(tp, pp, obs) and can(pp, dp, obs)
