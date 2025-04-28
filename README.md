# cmpe591-hw3-Ramazan Onur Acar
(I wanted to save my previous work before the extensions, so they are in the archieve folder if you are interested in the journey)

You can comment out test and train function calls in the main of the files.

**USAGE**

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python run_sac.py

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python homework3.py


## SAC PLOT 
![SAC](/sac.png)


## VPG PLOT
![VPG](/20k.png)


## Implementation Details

For the Vanilla Policy Gradient, I implemented several different logics for baesline and advantage. However, I couldn't make it learn in fewer than 10k episodes in my all trials. This version is my latest trial andI believe that It would have learned if I had tried with more episodes like 50k.

### Vanilla Policy Gradient (VPG)
- Implemented REINFORCE algorithm with a state-dependent baseline for variance reduction
- Used the high-level state representation
- For Baseline, simple MLP with two hidden layers for value function approximation
- Baseline discount factor: 0.9 for temporal difference updates
  - I aim to give more importance to recent rewards, improving responsiveness to recent experiences
  - Tried to help the value function adapt more quickly to the current policy's performance
- Normalized advantages to stabilize training
  - To Reduce variance in policy updates by standardizing the scale of advantages
  - To Prevent large policy shifts
- Trained for 20,000 episodes
- Continuous action space with actions scaled by delta=0.05


### Soft Actor-Critic (SAC)
- Off-policy algorithm with entropy maximization for exploration
- Twin Q-networks to mitigate overestimation bias
- Replay buffer size: 100,000 transitions
- Batch size: 256 for stable gradient estimates
- Entropy regularization coefficient (alpha): 0.2
- Discount factor (gamma): 0.99
- Learning rates: 3e-4 for both actor and critic networks
- Network architecture: [256, 256] hidden units with ReLU activations
- Experience collection: 1,000 steps before starting updates
- Trained for 3,000 episodes with intermediate model checkpoints every 250 episodes
- Deterministic policy for evaluation during testing

## Performance
- SAC achieved faster convergence and more stable training compared to VPG
- VPG required more episodes and didn't learn well for successful pushing strategies, maybe trying 30 - 40k episodes would make it successful.
- SAC demonstrated more sample efficiency and better final performance
