import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.distributions import Normal, Categorical

class SoftActorCritic:
    def __init__(self, model, lr=3e-4, gamma=0.99, alpha=0.2):
        self.model = model
        self.target_model = deepcopy(model)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Гиперпараметры
        self.gamma = gamma  # Фактор дисконтирования
        self.alpha = alpha  # Коэффициент энтропии
        
    def update(self, batch):
        """Обновление политики на батче данных"""
        states, actions, rewards, next_states, dones = batch
        
        # Критики
        _, _, _, q_value, _ = self.model(states)

        with torch.no_grad():
            _, _, _, next_q_value, _ = self.target_model(next_states)
        
        # Цель для Q-функции
        target_q = rewards + (1 - dones) * self.gamma * next_q_value
        
        # Loss критика
        critic_loss = F.mse_loss(q_value, target_q)
        
        # Актор
        (vol_mu, vol_sigma), order_logits, pos_logits, _, risk = self.model(states)
        volume_dist = Normal(vol_mu, vol_sigma)
        order_dist = Categorical(logits=order_logits)
        pos_dist = Categorical(logits=pos_logits)
        
        # Энтропийный член
        log_probs = volume_dist.log_prob(actions[0]) + \
                    order_dist.log_prob(actions[1]) + \
                    pos_dist.log_prob(actions[2])
        entropy = -self.alpha * log_probs.mean()
        
        # Advantage
        advantage = target_q - q_value
        
        # Loss актора
        actor_loss = -log_probs * advantage.detach() + entropy
        
        # Общий loss
        loss = critic_loss + actor_loss + risk_loss
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Обновление целевой сети
        self.soft_update_target()
        
    def soft_update_target(self, tau=0.005):
        """Мягкое обновление целевой сети"""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)