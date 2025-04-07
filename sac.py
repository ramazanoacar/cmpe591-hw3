import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

from my_hw3env import Hw3Env


class ReplayBuffer:
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=128):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1)
        )

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7).sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class SACAgent:
    def __init__(self,
                 state_dim=6,
                 action_dim=2,
                 hidden_dim=256,
                 gamma=0.99,
                 alpha=0.2,
                 lr=3e-4,
                 tau=0.005,
                 buffer_size=100000,
                 batch_size=256,
                 auto_entropy=True,
                 target_entropy=-2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy = auto_entropy
        self.target_entropy = target_entropy
        self.batch_size = batch_size

        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).float()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).float()
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).float()

        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).float()
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).float()
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        if self.auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.policy.forward(state_t)
        std = log_std.exp()
        if evaluate:
            z = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
        action = torch.tanh(z)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0, 0, 0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        # Update critic networks
        with torch.no_grad():
            next_action, next_logprob, _ = self.policy.sample(next_states_t)
            q1_next = self.q1_target(next_states_t, next_action)
            q2_next = self.q2_target(next_states_t, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_logprob
            target_q = rewards_t + (1 - dones_t) * self.gamma * q_next

        q1_pred = self.q1(states_t, actions_t)
        q2_pred = self.q2(states_t, actions_t)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # Update policy network
        action_sample, logprob_sample, _ = self.policy.sample(states_t)
        q1_val = self.q1(states_t, action_sample)
        q2_val = self.q2(states_t, action_sample)
        q_val = torch.min(q1_val, q2_val)
        policy_loss = (self.alpha * logprob_sample - q_val).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update temperature parameter
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (logprob_sample.detach() + self.target_entropy).mean())
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.)

        # Update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item()


def train():
    env = Hw3Env(render_mode="offscreen")
    state_dim = 6
    action_dim = 2

    agent = SACAgent(state_dim=state_dim,
                     action_dim=action_dim,
                     hidden_dim=256,
                     gamma=0.99,
                     alpha=0.2,
                     lr=3e-4,
                     tau=0.005,
                     buffer_size=100000,
                     batch_size=256,
                     auto_entropy=True,
                     target_entropy=-action_dim)

    max_episodes = 2000
    max_steps = 300
    rewards_log = []
    
    min_buffer_size = 5000
    updates_per_step = 10
    
    total_steps = 0
    
    for ep in range(max_episodes):
        state = env.reset()
        ep_reward = 0
        ep_steps = 0
        
        for t in range(max_steps):
            if total_steps < min_buffer_size:
                action = torch.FloatTensor(np.random.uniform(-1, 1, size=action_dim))
                action_np = action.numpy()
            else:
                action_np = agent.select_action(state, evaluate=False)
                action = torch.FloatTensor(action_np)
            
            next_state, reward, done, truncated = env.step(action)
            
            agent.replay_buffer.push(state, action_np, reward, next_state, done)
            
            if total_steps >= min_buffer_size:
                for _ in range(updates_per_step):
                    agent.update()
            
            state = next_state
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            
            if done or truncated:
                break
                
        rewards_log.append(ep_reward)
        
        print(f"Episode {ep+1} | steps={ep_steps} | reward={ep_reward:.2f} | alpha={agent.alpha:.3f}", flush=True)
        
        if (ep+1) % 250 == 0:
            np.save(f"sac_rewards_ep{ep+1}.npy", np.array(rewards_log))
            torch.save(agent.policy.state_dict(), f"sac_policy_ep{ep+1}.pth")
            print(f"[Checkpoint] Episode {ep+1} saved partial logs & model.")

    np.save("sac_rewards.npy", np.array(rewards_log))
    torch.save(agent.policy.state_dict(), "sac_policy.pth")
    print("Done training SAC. Saved final rewards and final policy.")

def test():
    env = Hw3Env(render_mode="offscreen")
    state_dim = 6
    action_dim = 2
    
    agent = SACAgent(state_dim=state_dim,
                     action_dim=action_dim,
                     hidden_dim=256,
                     gamma=0.99,
                     alpha=0.2,
                     lr=3e-4,
                     tau=0.005,
                     buffer_size=100000,
                     batch_size=256,
                     auto_entropy=True,
                     target_entropy=-action_dim)
    
    agent.policy.load_state_dict(torch.load("sac_policy.pth"))
    agent.policy.eval()
    
    # Test for 10 episodes
    num_episodes = 10
    rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        if state is None:
            print(f"Environment reset returned None on episode {ep+1}")
            continue
            
        done = False
        episode_reward = 0
        
        while not done:
            action_np = agent.select_action(state, evaluate=True)
            action = torch.FloatTensor(action_np)
            
            next_state, reward, done, truncated = env.step(action)
            done = done or truncated
            
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
        print(f"Test Episode {ep+1}, Reward: {episode_reward:.2f}")
    
    mean_reward = sum(rewards) / len(rewards) if rewards else 0
    print(f"Average reward over {len(rewards)} episodes: {mean_reward:.2f}")
    
    return mean_reward

if __name__ == "__main__":
    # train()
    test()
