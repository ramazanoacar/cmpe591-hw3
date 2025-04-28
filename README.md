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

### Vanilla Policy Gradient (VPG)
- Implemented REINFORCE algorithm with a state-dependent baseline for variance reduction
- Used the high-level state representation
- For Baseline, simple MLP with two hidden layers for value function approximation
- Baseline discount factor: 0.9 for temporal difference updates
  - This gives more importance to recent rewards, improving responsiveness to recent experiences
  - Helps the value function adapt more quickly to the current policy's performance
- Normalized advantages to stabilize training
  - Reduces variance in policy updates by standardizing the scale of advantages
  - Prevents large policy shifts from outlier advantage values
  - Makes training less sensitive to reward scaling and more consistent across environments
- Policy network: Multi-layer perceptron with tanh activations
- Trained for 20,000 episodes to ensure convergence
- Continuous action space with actions scaled by delta=0.05


### Soft Actor-Critic (SAC)
- Off-policy algorithm with entropy maximization for exploration
- Twin Q-networks to mitigate overestimation bias
- Replay buffer size: 100,000 transitions
- Batch size: 256 for stable gradient estimates
- Entropy regularization coefficient (alpha): 0.2
- Discount factor (gamma): 0.99
- Polyak averaging coefficient: 0.995 for target network updates
- Learning rates: 3e-4 for both actor and critic networks
- Network architecture: [256, 256] hidden units with ReLU activations
- Experience collection: 1,000 steps before starting updates
- Trained for 3,000 episodes with intermediate model checkpoints every 250 episodes
- Deterministic policy for evaluation during testing

## Performance
- SAC achieved faster convergence and more stable training compared to VPG
- VPG required more episodes and didn't learn well for successful pushing strategies, maybe trying 30 - 40k episodes would make it successful.
- SAC demonstrated more sample efficiency and better final performance
