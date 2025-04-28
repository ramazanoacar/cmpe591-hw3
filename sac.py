import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Neural network for the Q-function (state-action value)
class QFunction(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hidden_dims=[256, 256]):
        super(QFunction, self).__init__()
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q_net(x)

# Neural network for the actor (policy)
class Actor(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hidden_dims=[256, 256], log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_dim = act_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(hidden_dims[1], act_dim)
        self.log_std_layer = nn.Linear(hidden_dims[1], act_dim)
        
    def forward(self, obs):
        net_out = self.net(obs)
        mean = self.mean_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mean, std
        
    def sample(self, obs):
        mean, std = self.forward(obs)
        normal = torch.distributions.Normal(mean, std)
        
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# SAC Agent
class SACAgent:
    def __init__(
        self,
        obs_dim=6,
        act_dim=2,
        hidden_dims=[256, 256],
        buffer_size=100000,
        batch_size=256,
        alpha=0.2,
        gamma=0.99,
        polyak=0.995,
        actor_lr=3e-4,
        q_lr=3e-4,
        device="cpu"
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = device
        
        self.actor = Actor(obs_dim, act_dim, hidden_dims).to(device)
        self.q1 = QFunction(obs_dim, act_dim, hidden_dims).to(device)
        self.q2 = QFunction(obs_dim, act_dim, hidden_dims).to(device)
        
        self.q1_target = QFunction(obs_dim, act_dim, hidden_dims).to(device)
        self.q2_target = QFunction(obs_dim, act_dim, hidden_dims).to(device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=q_lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.all_episode_rewards = []
        self.episode_count = 0
        
    def decide_action(self, state, evaluate=False):
        # Sample action from policy
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)
                
        return action[0]
    
    def add_experience(self, state, action, reward, next_state, done):
        # Convert any tensors to numpy arrays
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update_model(self):
        # Update model using batch actor-critic algorithm
        if len(self.replay_buffer) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next
            
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        new_actions, log_probs = self.actor.sample(states)
        
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self._update_target_networks()
        
    def _update_target_networks(self):
        """Soft update target networks"""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)
            
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)
    
    def add_reward(self, episode_reward):
        """Store episode reward"""
        self.all_episode_rewards.append(episode_reward)
        self.episode_count += 1
        
        # Save progress every 250 episodes
        if self.episode_count % 250 == 0:
            self.save_progress()
    
    def save_progress(self):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'episode_count': self.episode_count,
        }, f'sac_checkpoint_{self.episode_count}.pt')
        
        np.save(f'sac_rewards_{self.episode_count}.npy', np.array(self.all_episode_rewards)) 