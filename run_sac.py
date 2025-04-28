import torch
import numpy as np
import matplotlib.pyplot as plt

from sac import SACAgent
from homework3 import Hw3Env

def train():
    env = Hw3Env(render_mode="offscreen")
    
    agent = SACAgent(
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
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Training parameters
    num_episodes = 3000
    min_steps_before_update = 1000
    total_steps = 0
    rewards = []
    
    for episode in range(num_episodes):
        env.reset()
        state = env.high_level_state()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.decide_action(state)
            
            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            
            agent.add_experience(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if total_steps > min_steps_before_update:
                agent.update_model()
        
        rewards.append(episode_reward)
        agent.add_reward(episode_reward)
        
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")
        
        if (episode + 1) % 250 == 0:
            np.save(f'sac_rewards_{episode+1}.npy', np.array(rewards))
            
            plt.figure(figsize=(10, 5))
            plt.plot(rewards)
            plt.title('SAC Training Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(f'sac_rewards_{episode+1}.png')
            plt.close()
            
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'q1_state_dict': agent.q1.state_dict(),
                'q2_state_dict': agent.q2.state_dict(),
                'q1_target_state_dict': agent.q1_target.state_dict(),
                'q2_target_state_dict': agent.q2_target.state_dict(),
            }, f'sac_model_{episode+1}.pt')
    
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'q1_state_dict': agent.q1.state_dict(),
        'q2_state_dict': agent.q2.state_dict(),
        'q1_target_state_dict': agent.q1_target.state_dict(),
        'q2_target_state_dict': agent.q2_target.state_dict(),
    }, 'sac_final_model.pt')
    
    np.save('sac_rewards_final.npy', np.array(rewards))
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('SAC Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('sac_rewards_final.png')
    plt.close()
    
    return agent

def test(model_path='sac_final_model.pt', num_episodes=10, render=True):
    render_mode = "human" if render else "offscreen"
    env = Hw3Env(render_mode=render_mode)
    
    agent = SACAgent(
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
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    checkpoint = torch.load(model_path, map_location=agent.device)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.q1.load_state_dict(checkpoint['q1_state_dict'])
    agent.q2.load_state_dict(checkpoint['q2_state_dict'])
    agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
    agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
    
    test_rewards = []
    
    for episode in range(num_episodes):
        env.reset()
        state = env.high_level_state()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.decide_action(state)
            
            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            
            state = next_state
            episode_reward += reward
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode}, Reward: {episode_reward:.2f}")
    
    avg_reward = np.mean(test_rewards)
    print(f"Average Test Reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    return test_rewards


if __name__ == "__main__":
    # train()
    test()