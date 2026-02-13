import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class NestedTaxiAgent(nn.Module):
    """
    Arquitectura de Nested Learning (Slow-Fast Networks) para AutoMind-Agent.
    Diseñada para mitigar el olvido catastrófico en navegación autónoma.
    """
    def __init__(self, state_dim, action_dim, lstm_hidden=128, tau=0.995):
        super(NestedTaxiAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau  # Coeficiente EMA para la Slow Network

        # --- Fast Network (Exploración Adaptativa) ---
        self.fast_feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fast_lstm = nn.LSTM(128, lstm_hidden, batch_first=True)
        self.fast_actor = nn.Linear(lstm_hidden, action_dim)
        self.fast_critic = nn.Linear(lstm_hidden, 1)

        # --- Slow Network (Global Policy Buffer - Conocimiento Persistente) ---
        # Inicializada como copia de la Fast Network
        self.slow_feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.slow_lstm = nn.LSTM(128, lstm_hidden, batch_first=True)
        self.slow_actor = nn.Linear(lstm_hidden, action_dim)
        self.slow_critic = nn.Linear(lstm_hidden, 1)

        # Congelar los parámetros de la Slow Network inicialmente
        for param in self.slow_parameters():
            param.requires_grad = False

    def slow_parameters(self):
        """Generador para los parámetros de la Slow Network."""
        return (list(self.slow_feature_extractor.parameters()) + 
                list(self.slow_lstm.parameters()) + 
                list(self.slow_actor.parameters()) + 
                list(self.slow_critic.parameters()))

    def update_slow_network(self):
        """
        Actualiza la Slow Network usando una Media Móvil Exponencial (EMA).
        Conexión directa con la filosofía de consolidación de memoria a largo plazo.
        """
        with torch.no_grad():
            for slow_p, fast_p in zip(self.slow_parameters(), self.parameters()):
                if fast_p.requires_grad: # Solo actualizar desde la red activa
                    slow_p.data.copy_(self.tau * slow_p.data + (1.0 - self.tau) * fast_p.data)

    def calculate_surprise(self, state, reward, next_state, done):
        """
        Mecanismo de Selección: Surprise-based Learning.
        Calcula el TD-error como métrica de sorpresa para el filtrado de memoria episódica.
        """
        with torch.no_grad():
            v_s = self.fast_critic(self.fast_feature_extractor(state))
            v_s_next = self.fast_critic(self.fast_feature_extractor(next_state))
            surprise = torch.abs(reward + (1 - done) * 0.99 * v_s_next - v_s)
        return surprise.item()

    def forward(self, x, hidden=None, use_slow=False):
        """
        Forward pass híbrido. Por defecto usa la Fast Network para acción reactiva.
        """
        net = self.slow_feature_extractor if use_slow else self.fast_feature_extractor
        lstm = self.slow_lstm if use_slow else self.fast_lstm
        actor = self.slow_actor if use_slow else self.fast_actor
        
        features = net(x)
        features, hidden = lstm(features, hidden)
        action_probs = F.softmax(actor(features), dim=-1)
        
        return action_probs, hidden

def get_torch_threads_optimized():
    """
    Configura PyTorch para usar todos los hilos del i9-10900KF (20 hilos).
    """
    total_threads = 20
    torch.set_num_threads(total_threads)
    torch.set_num_interop_threads(total_threads)
    print(f"🚀 PyTorch optimizado para {total_threads} hilos (i9-10900KF).")

if __name__ == "__main__":
    get_torch_threads_optimized()
    # Demo de inicialización
    agent = NestedTaxiAgent(state_dim=50, action_dim=6)
    print("✅ NestedTaxiAgent inicializado con éxito.")
