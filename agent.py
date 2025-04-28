import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

from model import VPG

class Agent():
    def __init__(self, lr=3e-4, baseline_discount=0.9, normalize_advantages=True):
        self.model = VPG()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.trajectory_states = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
        
        self.all_episode_rewards = []
        self.episode_count = 0
        
        # Discount factor for baseline calculation
        self.baseline_discount = baseline_discount
        
        # Whether to normalize advantages
        self.normalize_advantages = normalize_advantages
        
    def decide_action(self, state):
        """Step 1 of REINFORCE: Sample trajectory τ from π_θ(a_t|s_t)"""
        state = torch.FloatTensor(state)
        
        mean, log_std = self.model(state).chunk(2, dim=-1)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        
        self.trajectory_states.append(state)
        self.trajectory_actions.append(action)
        
        return action
    
    def add_reward(self, reward):
        self.trajectory_rewards.append(reward)
    
    def _compute_rewards_to_go(self, rewards):
        # Compute rewards-to-go for each timestep: sum of all future rewards
        rewards = torch.FloatTensor(rewards)
        rewards_to_go = torch.zeros_like(rewards)
        
        for t in range(len(rewards)):
            rewards_to_go[t] = rewards[t:].sum()
            
        return rewards_to_go
    
    def _compute_discounted_baseline(self, rewards_to_go):
        # Compute time-dependent baseline with higher weight for recent rewards
        if len(rewards_to_go) == 0:
            return 0.0
        
        discount_weights = torch.tensor([self.baseline_discount ** (len(rewards_to_go) - i - 1) 
                                        for i in range(len(rewards_to_go))])
        
        # Normalize weights to sum to 1
        discount_weights = discount_weights / discount_weights.sum()
        
        # Compute weighted average of rewards-to-go
        weighted_baseline = (rewards_to_go * discount_weights).sum()
        
        return weighted_baseline
    
    def _normalize_advantages(self, advantages):
        if len(advantages) <= 1:
            return advantages
        
        mean = advantages.mean()
        std = advantages.std()
        
        std = torch.maximum(std, torch.tensor(1e-8))
        
        normalized_advantages = (advantages - mean) / std
        
        return normalized_advantages
    
    def update_model(self):
        # Steps 2 and 3 of REINFORCE with baseline and rewards-to-go
        self.episode_count += 1
        
        states = torch.stack(self.trajectory_states)
        actions = torch.stack(self.trajectory_actions)
        
        rewards_to_go = self._compute_rewards_to_go(self.trajectory_rewards)
        
        episode_total_reward = sum(self.trajectory_rewards)
        self.all_episode_rewards.append(episode_total_reward)
        
        baseline = self._compute_discounted_baseline(rewards_to_go)
        
        advantages = rewards_to_go - baseline
        
        if self.normalize_advantages:
            advantages = self._normalize_advantages(advantages)
        
        mean, log_std = self.model(states).chunk(2, dim=-1)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)  # Sum over action dimensions
        
        policy_gradient = -(log_probs * advantages).mean()
        
        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()
        
        # Save progress every 250 episodes
        if self.episode_count % 250 == 0:
            self.save_progress()
        
        self.trajectory_states = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
    
    def save_progress(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'baseline_discount': self.baseline_discount
        }, f'model_checkpoint_{self.episode_count}.pt')
        
        np.save(f'rewards_{self.episode_count}.npy', np.array(self.all_episode_rewards))
        
