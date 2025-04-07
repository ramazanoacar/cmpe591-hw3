
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from _homework3 import Hw3Env

from model import VPG

def sample_action_and_logprob(policy: VPG, state: torch.Tensor):
    """
    Input:
      state: [batch, obs_dim], e.g. [batch, 6]
    Output:
      (action, logprob)
    """
    out = policy(state)
    mean, log_std = out[:, :2], out[:, 2:]
    std = torch.exp(log_std)

    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    logprob = dist.log_prob(action).sum(dim=-1, keepdim=True)
    return action, logprob

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()

    def append(self, state, action, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)

def collector_proc(policy, queue, is_collecting, is_finished, device, w_id):
    env = Hw3Env(render_mode="offscreen")
    env._max_timesteps = 200

    policy = policy.to(device)
    policy.eval()

    while not is_finished.is_set():
        if not is_collecting.is_set():
            is_collecting.wait()
            if is_finished.is_set():
                break

        state_np = env.reset()
        if state_np is None:
            print(f"[Worker {w_id}] Env reset returned None!")
            break

        state_t = torch.tensor(state_np, dtype=torch.float32, device=device)
        done = False

        while not done:
            action_t, logprob_t = sample_action_and_logprob(policy, state_t.unsqueeze(0))
            action_np = action_t[0].detach().cpu().numpy()
            logprob_np = logprob_t[0].detach().cpu().numpy()

            next_state_np, reward, terminal, truncated = env.step(torch.tensor(action_np))
            done = terminal or truncated

            queue.put((
                state_t.cpu().numpy(),
                action_np,
                logprob_np,
                reward,
                done
            ))

            if not done:
                state_t = torch.tensor(next_state_np, dtype=torch.float32, device=device)

            if is_finished.is_set():
                break

def train():
    mp.set_start_method("spawn", force=True)
    device = torch.device("cpu")

    policy = VPG(obs_dim=6, act_dim=2, hl=[128, 128])
    policy.share_memory()

    queue = mp.Queue(maxsize=10000)

    is_collecting = mp.Event()
    is_collecting.set()
    is_finished = mp.Event()

    # spawn worker processes
    workers = []
    num_workers = 2
    for w_id in range(num_workers):
        p = mp.Process(
            target=collector_proc,
            args=(policy, queue, is_collecting, is_finished, device, w_id),
        )
        p.start()
        workers.append(p)

    # Simple REINFORCE loop
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    memory = Memory()

    max_episodes = 5000
    gamma = 0.99
    episode_count = 0
    all_rewards = []

    while episode_count < max_episodes:
        memory.clear()
        ep_reward = 0.0

        while True:
            s, a, lp, r, d = queue.get()
            memory.append(s, a, lp, r, d)
            ep_reward += r
            if d:
                break

        returns = []
        G = 0.0
        for t in reversed(range(len(memory.states))):
            G = memory.rewards[t] + gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)
        returns_t = torch.from_numpy(returns).unsqueeze(-1)

        states_arr = np.array(memory.states, dtype=np.float32)
        actions_arr = np.array(memory.actions, dtype=np.float32)
        logprobs_arr = np.array(memory.logprobs, dtype=np.float32)

        states_t = torch.from_numpy(states_arr)
        actions_t = torch.from_numpy(actions_arr)
        logprobs_t = torch.from_numpy(logprobs_arr)

        out = policy(states_t)
        mean, log_std = out[:, :2], out[:, 2:]
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        new_logprobs_t = dist.log_prob(actions_t).sum(dim=-1, keepdim=True)

        # REINFORCE loss = - E[G * log pi(a|s)]
        loss = - (returns_t * new_logprobs_t).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_count += 1
        all_rewards.append(ep_reward)
        print(f"Episode {episode_count}, reward={ep_reward:.2f}")

    print("Finishing up, please wait.")
    is_finished.set()
    for w in workers:
        w.join()

    torch.save(policy.state_dict(), "vpg_model.pth")
    np.save("train_rewards.npy", np.array(all_rewards))
    print("Done. Saved final policy and training rewards.")


def test():
    env = Hw3Env(render_mode="offscreen")
    env._max_timesteps = 200
    
    policy = VPG(obs_dim=6, act_dim=2, hl=[128, 128])
    policy.load_state_dict(torch.load("vpg_model.pth"))
    policy.eval()
    
    num_episodes = 10
    rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        if state is None:
            print(f"Environment reset returned None on episode {ep+1}")
            continue
            
        state_t = torch.tensor(state, dtype=torch.float32)
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                action_t, _ = sample_action_and_logprob(policy, state_t.unsqueeze(0))
            
            action_np = action_t[0].cpu().numpy()
            
            next_state, reward, terminal, truncated = env.step(torch.tensor(action_np))
            done = terminal or truncated
            
            episode_reward += reward
            
            if not done:
                state_t = torch.tensor(next_state, dtype=torch.float32)
        
        rewards.append(episode_reward)
        print(f"Test Episode {ep+1}, Reward: {episode_reward:.2f}")
    
    mean_reward = sum(rewards) / len(rewards) if rewards else 0
    print(f"Average reward over {len(rewards)} episodes: {mean_reward:.2f}")
    
    return mean_reward

if __name__ == "__main__":
    # train()
    test()
