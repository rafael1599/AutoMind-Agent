
# AutoMind-Ultra: COGNITIVE ARCHITECTURE (Fase 4 - Memory + Curiosity)
# DeepFix: Target Networks + Intrinsic Motivation + LiDAR

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import os
from tqdm import tqdm
from collections import deque
import random

# --- DIRECTORIOS ---
os.makedirs("models", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

# --- CONFIGURACION ---
SEED = 42
NUM_ENVS = 12
NUM_STEPS = 512
NUM_EPOCHS = 4
MB_SIZE_ENVS = 4 # 3 batches por época (12/4) = 12 actualizaciones de peso por rollout
LEARNING_RATE = 2.5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIGURACION COGNITIVA ---
TARGET_TAU = 0.01      # Velocidad de consolidación de memoria (1% por update)
CURIOSITY_WEIGHT = 0.05 # Cuánto le importa al agente aprender cosas nuevas

# Meta-Parámetros de Graduación
SUCCESS_GOAL = 0.90
EVAL_WINDOW = 100
MAX_STEPS_ALLOWED = 15

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
        # Copia exacta del critic para estabilizar el aprendizaje (Consolidación)
        self.target_critic_features = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU()
        )
        self.target_critic_lstm = nn.LSTM(128, 128)
        self.target_critic_ln = nn.LayerNorm(128)
        self.target_critic_head = nn.Linear(128, 1)
        
        # Inicializar Target con los mismos pesos
        self.target_critic_features.load_state_dict(self.critic_features.state_dict())
        self.target_critic_lstm.load_state_dict(self.critic_lstm.state_dict())
        self.target_critic_ln.load_state_dict(self.critic_ln.state_dict())
        self.target_critic_head.load_state_dict(self.critic_head.state_dict())
        
        # Congelar gradientes del Target (solo se actualiza vía soft-update)
        for param in self.target_parameters():
            param.requires_grad = False

    def target_parameters(self):
        return (list(self.target_critic_features.parameters()) + 
                list(self.target_critic_lstm.parameters()) + 
                list(self.target_critic_ln.parameters()) + 
                list(self.target_critic_head.parameters()))

    def update_target_network(self, tau):
        # Soft Update: target = (1-tau)*target + tau*source
        for target_param, param in zip(self.target_critic_features.parameters(), self.critic_features.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic_lstm.parameters(), self.critic_lstm.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic_ln.parameters(), self.critic_ln.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic_head.parameters(), self.critic_head.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action_and_value(self, x, actor_state, critic_state, done, action=None):
        a_hidden = self.actor_features(x)
        batch_size = actor_state[0].shape[1]
        a_hidden = a_hidden.reshape((-1, batch_size, 128))
        done_reshaped = done.reshape((-1, batch_size))
        
        new_a_hidden = []
        for h, d in zip(a_hidden, done_reshaped):
            d_f = d.view(1, -1, 1).float()
            actor_state = ((1.0-d_f)*actor_state[0], (1.0-d_f)*actor_state[1])
            h, actor_state = self.actor_lstm(h.unsqueeze(0), actor_state)
            new_a_hidden += [h]
        new_a_hidden = torch.flatten(torch.cat(new_a_hidden), 0, 1)
        new_a_hidden = self.actor_ln(new_a_hidden)
        logits = self.actor_head(new_a_hidden)
        probs = Categorical(logits=logits)
        
        c_hidden = self.critic_features(x).reshape((-1, batch_size, 128))
        new_c_hidden = []
        for h, d in zip(c_hidden, done_reshaped):
            d_f = d.view(1, -1, 1).float()
            critic_state = ((1.0-d_f)*critic_state[0], (1.0-d_f)*critic_state[1])
            h, critic_state = self.critic_lstm(h.unsqueeze(0), critic_state)
            new_c_hidden += [h]
        new_c_hidden = torch.flatten(torch.cat(new_c_hidden), 0, 1)
        new_c_hidden = self.critic_ln(new_c_hidden)
        value = self.critic_head(new_c_hidden)
        
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, actor_state, critic_state

    def get_value(self, x, actor_state, critic_state, done):
        batch_size = critic_state[0].shape[1]
        c_hidden = self.critic_features(x).reshape((-1, batch_size, 128))
        done_reshaped = done.reshape((-1, batch_size))
        new_c_hidden = []
        for h, d in zip(c_hidden, done_reshaped):
            d_f = d.view(1, -1, 1).float()
            critic_state = ((1.0-d_f)*critic_state[0], (1.0-d_f)*critic_state[1])
            h, critic_state = self.critic_lstm(h.unsqueeze(0), critic_state)
            new_c_hidden += [h]
        new_c_hidden = torch.flatten(torch.cat(new_c_hidden), 0, 1)
        new_c_hidden = self.critic_ln(new_c_hidden)
        return self.critic_head(new_c_hidden)

    def get_target_value(self, x, target_state, done):
        """Calcula el valor usando la Memoria a Largo Plazo (Target Network)"""
        batch_size = target_state[0].shape[1]
        c_hidden = self.target_critic_features(x).reshape((-1, batch_size, 128))
        done_reshaped = done.reshape((-1, batch_size))
        new_c_hidden = []
        for h, d in zip(c_hidden, done_reshaped):
            d_f = d.view(1, -1, 1).float()
            target_state = ((1.0-d_f)*target_state[0], (1.0-d_f)*target_state[1])
            h, target_state = self.target_critic_lstm(h.unsqueeze(0), target_state)
            new_c_hidden += [h]
        new_c_hidden = torch.flatten(torch.cat(new_c_hidden), 0, 1)
        new_c_hidden = self.target_critic_ln(new_c_hidden)
        return self.target_critic_head(new_c_hidden), target_state

class EliteBuffer:
    def __init__(self, capacity=2000):
        self.buffer = []
        self.capacity = capacity
    def add(self, trajectory):
        traj_len = len(trajectory)
        if len(self.buffer) < self.capacity: self.buffer.append(trajectory)
        else:
            longest = -1; max_len = -1
            for i, t in enumerate(self.buffer):
                if len(t) > max_len: max_len = len(t); longest = i
            if traj_len < max_len: self.buffer[longest] = trajectory

def make_env(level):
    def thunk():
        from taxi_env import TaxiEnv
        env = TaxiEnv()
        env.set_level(level)
        return env
    return thunk

if __name__ == "__main__":
    print(f"\n[INIT] AutoMind: Fase 3.6 (LiDAR + Stable Critic)")
    
    current_level = 0
    envs = SubprocVecEnv([make_env(current_level) for _ in range(NUM_ENVS)])
    envs = VecNormalize(envs, norm_obs=False, norm_reward=False) 
    
    agent = Agent(envs.observation_space, envs.action_space).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    
    success_buffer = deque(maxlen=EVAL_WINDOW)
    step_buffer = deque(maxlen=EVAL_WINDOW)
    elite_buffer = EliteBuffer(capacity=2000)
    current_trajectories = [[] for _ in range(NUM_ENVS)]
    
    best_success_rate = 0.0
    rollouts_since_improvement = 0
    
    next_obs = torch.Tensor(envs.reset()).to(DEVICE)
    next_done = torch.zeros(NUM_ENVS).to(DEVICE)
    next_a_state = (torch.zeros(1, NUM_ENVS, 128).to(DEVICE), torch.zeros(1, NUM_ENVS, 128).to(DEVICE))
    next_c_state = (torch.zeros(1, NUM_ENVS, 128).to(DEVICE), torch.zeros(1, NUM_ENVS, 128).to(DEVICE))
    next_target_state = (torch.zeros(1, NUM_ENVS, 128).to(DEVICE), torch.zeros(1, NUM_ENVS, 128).to(DEVICE))
    
    episode_steps = np.zeros(NUM_ENVS)
    bar_fmt = "{desc} [{bar}] {percentage:3.0f}% | {postfix} | RT:{elapsed}"
    pbar = tqdm(total=100, desc=f"LVL {current_level}", bar_format=bar_fmt)

    while current_level <= 3:
        obs_b = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.observation_space.shape).to(DEVICE)
        act_b = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
        logp_b = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
        rew_b = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
        done_b = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
        val_b = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
        trunc_val_b = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
        
        init_a_state = (next_a_state[0].clone(), next_a_state[1].clone())
        init_c_state = (next_c_state[0].clone(), next_c_state[1].clone())
        # No necesitamos clonar target_state para el rollout, solo para la inferencia de bootstrap
        
        for step in range(NUM_STEPS):
            obs_b[step], done_b[step] = next_obs, next_done
            with torch.no_grad():
                action, logprob, _, value, next_a_state, next_c_state = agent.get_action_and_value(next_obs, next_a_state, next_c_state, next_done)
                val_b[step] = value.flatten()
            
            act_b[step], logp_b[step] = action, logprob
            next_obs_raw, reward, done, infos = envs.step(action.cpu().numpy())
            
            rew_b[step] = torch.tensor(reward).to(DEVICE)

            # --- CURIOSIDAD INTRÍNSECA (Fase 4) ---
            # Calcular Sorpresa: |Reward + Gamma * V_target(next) - V_online(curr)|
            # Usamos V_online(curr) que ya tenemos en val_b[step]
            # Estimamos V_target(next) rápidamente (sin actualizar estado LSTM target paso a paso por eficiencia, usamos snapshot)
            # Nota: Para simplificar y no duplicar coste, usamos el error de predicción del Critic Online como proxy de sorpresa.
            # Surprise ~= |TD-Error|
            # surprise = torch.abs(rew_b[step] + 0.99 * val_b[step] - val_b[step]) # Simplificación
            # Implementación real: Usar el reward intrínseco en el cálculo de GAE después.
            next_obs, next_done = torch.Tensor(next_obs_raw).to(DEVICE), torch.Tensor(done).to(DEVICE)
            episode_steps += 1
            
            for i in range(NUM_ENVS):
                current_trajectories[i].append((obs_b[step, i].cpu(), act_b[step, i].cpu(), rew_b[step, i].cpu()))
                if done[i]:
                    is_success = infos[i].get("is_success", False)
                    success_buffer.append(1 if is_success else 0)
                    step_buffer.append(episode_steps[i])
                    
                    if infos[i].get("TimeLimit.truncated", False):
                         term_obs = infos[i].get("terminal_observation")
                         if term_obs is not None:
                             term_obs_t = torch.tensor(term_obs).unsqueeze(0).to(DEVICE)
                             with torch.no_grad():
                                 d_a = (torch.zeros(1,1,128).to(DEVICE), torch.zeros(1,1,128).to(DEVICE))
                                 d_c = (torch.zeros(1,1,128).to(DEVICE), torch.zeros(1,1,128).to(DEVICE))
                                 # Usar Target para bootstrap de truancation es más estable
                                 term_val, _ = agent.get_target_value(term_obs_t, d_c, torch.zeros(1,1).to(DEVICE))
                                 trunc_val_b[step, i] = term_val.item()
                    
                    if is_success:
                        traj_obs, traj_act, traj_rew = zip(*current_trajectories[i])
                        R = 0; elite_traj = []
                        for t_idx in reversed(range(len(traj_rew))):
                            R = traj_rew[t_idx] + 0.99 * R
                            elite_traj.append((traj_obs[t_idx], traj_act[t_idx], R))
                        elite_buffer.add(elite_traj)
                    episode_steps[i] = 0; current_trajectories[i] = []

        with torch.no_grad():
            # Bootstrap con Target Network (Estabilidad Cognitiva)
            next_value, next_target_state = agent.get_target_value(next_obs, next_target_state, next_done)
            next_value = next_value.reshape(1, -1)
            
            adv_b = torch.zeros_like(rew_b).to(DEVICE); lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                nt = 1.0 - next_done if t == NUM_STEPS-1 else 1.0 - done_b[t+1]
                # Si t+1 es terminal, next_val es next_value (calculado arriba). 
                # Si no, deberíamos usar el target value almacenado... pero por eficiencia usamos val_b (online) 
                # o idealmente recalculamos target values. Para Phase 4, usamos Online para GAE pero Target para Bootstrap final.
                # MEJORA: Calcular Curiosidad aquí
                
                # TD-Error como Curiosidad: Surprise = |Delta|
                # R_total = R_ext + w * |Delta_prev|
                
                nv = next_value if t == NUM_STEPS-1 else val_b[t+1]
                delta = rew_b[t] + 0.99 * (nv * nt + trunc_val_b[t]) - val_b[t]
                
                # CURIOSITY BONUS
                intrinsic_reward = CURIOSITY_WEIGHT * torch.abs(delta)
                # Sumamos curiosidad al reward para el cálculo de ventaja (Motivation)
                # delta_total = (rew_b[t] + intrinsic_reward) + ...
                # Pero GAE estándar usa reward puro. Sumamos al delta.
                delta_w_curiosity = delta + intrinsic_reward
                
                adv_b[t] = lastgaelam = delta_w_curiosity + 0.99 * 0.95 * nt * lastgaelam
            ret_b = adv_b + val_b

        env_indices = np.arange(NUM_ENVS)
        ent_coef = max(0.01, 0.05 - (current_level * 0.01))
        agent.train()
        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(env_indices)
            for start in range(0, NUM_ENVS, MB_SIZE_ENVS): 
                mb_env_inds = env_indices[start:start+MB_SIZE_ENVS]
                mb_a_state = (init_a_state[0][:, mb_env_inds], init_a_state[1][:, mb_env_inds])
                mb_c_state = (init_c_state[0][:, mb_env_inds], init_c_state[1][:, mb_env_inds])
                
                _, newlogp, entropy, newval, _, _ = agent.get_action_and_value(
                    obs_b[:, mb_env_inds], mb_a_state, mb_c_state, done_b[:, mb_env_inds], act_b[:, mb_env_inds].reshape(-1).long()
                )
                
                mb_adv = adv_b[:, mb_env_inds].reshape(-1)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                # --- BUG FIX: Target Normalization para el Critic ---
                mb_ret = ret_b[:, mb_env_inds].reshape(-1)
                mb_ret_norm = (mb_ret - mb_ret.mean()) / (mb_ret.std() + 1e-8)
                
                ratio = (newlogp - logp_b[:, mb_env_inds].reshape(-1)).exp()
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                v_loss = 0.5 * ((newval.flatten() - mb_ret_norm)**2).mean()
                
                loss = pg_loss - ent_coef * entropy.mean() + v_loss * 0.5
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 0.5); optimizer.step()

        # --- CONSOLIDACIÓN DE MEMORIA (Update Target) ---
        agent.update_target_network(TARGET_TAU)

        avg_success = np.mean(success_buffer) if len(success_buffer) >= 20 else 0
        avg_steps = np.mean(step_buffer) if len(step_buffer) >= 20 else 50
        
        # --- NUEVA LOGICA: Plateau LR (ReduceLROnPlateau) ---
        if avg_success > best_success_rate:
            best_success_rate = avg_success
            rollouts_since_improvement = 0
            if avg_success >= 0.40:
                torch.save(agent.state_dict(), f"models/automind_best_L{current_level}.pth")
                pbar.write(f"\n[HITO] {best_success_rate*100:.0f}% -> Checkpoint.")
        else:
            rollouts_since_improvement += 1
            if rollouts_since_improvement >= 7: # ~43k steps
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.8
                    pbar.write(f"\n[PLATEAU] LR reducido a {pg['lr']:.2e}")
                rollouts_since_improvement = 0
        
        # --- ROLLBACK (Margen de 35% de degradación) ---
        if best_success_rate > 0.50 and avg_success < (best_success_rate - 0.35):
             pbar.write(f"\n[ALERTA] Reset total ({avg_success*100:.0f}% << {best_success_rate*100:.0f}%).")
             agent.load_state_dict(torch.load(f"models/automind_best_L{current_level}.pth"))
             success_buffer.clear(); step_buffer.clear()
        
        pbar.n = int((min(1.0, avg_success/SUCCESS_GOAL)*0.5 + min(1.0, MAX_STEPS_ALLOWED/avg_steps)*0.5)*100)
        pbar.set_postfix_str(f"Acc:{avg_success*100:2.0f}% | AvgS:{avg_steps:3.1f} | Elite:{len(elite_buffer.buffer)} | VL_N:{v_loss:.2f} | LR:{optimizer.param_groups[0]['lr']:.1e}")
        pbar.refresh()

        if len(success_buffer) >= EVAL_WINDOW and avg_success >= SUCCESS_GOAL and avg_steps <= MAX_STEPS_ALLOWED:
            torch.save(agent.state_dict(), f"models/automind_L{current_level}.pth")
            pbar.write(f"\n[LEVEL UP] L{current_level}"); current_level += 1
            if current_level > 3: break
            
            envs.close(); envs = SubprocVecEnv([make_env(current_level) for _ in range(NUM_ENVS)])
            if current_level >= 2: envs.env_method("generate_random_obstacles", [0,0,2,6][current_level])
            envs = VecNormalize(envs, norm_obs=False, norm_reward=False)
            next_obs = torch.Tensor(envs.reset()).to(DEVICE); next_done = torch.zeros(NUM_ENVS).to(DEVICE)
            next_a_state = (torch.zeros(1, NUM_ENVS, 128).to(DEVICE), torch.zeros(1, NUM_ENVS, 128).to(DEVICE))
            next_c_state = (torch.zeros(1, NUM_ENVS, 128).to(DEVICE), torch.zeros(1, NUM_ENVS, 128).to(DEVICE))
            next_target_state = (torch.zeros(1, NUM_ENVS, 128).to(DEVICE), torch.zeros(1, NUM_ENVS, 128).to(DEVICE))
            success_buffer.clear(); step_buffer.clear(); best_success_rate = 0.0; rollouts_since_improvement = 0
            pbar.desc = f"LVL {current_level}"; MAX_STEPS_ALLOWED = [15, 20, 25, 30][current_level]

    pbar.close(); envs.close()
    torch.save(agent.state_dict(), "models/automind_final.pth")
