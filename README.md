# cmpe591-hw3-Ramazan Onur Acar
(I wanted to save my previous work before the extensions, so they are in the archieve folder if you are interested in the journey)
In this homework, I initially trained SAC and VPG with the previous reward function, the one in the "_homework3.py" because of a misunderstanding.

Later, I did with the newest reward function. However, I faced an issue: Because of some worker issue, my new version of VPG died just before saving the rewards. 

**IMPORTANT**

Therefore, I will submit the first version of VPG I made, my VPG reward calculations below will have the plot and model from the "_homework3.py"s environment and my SAC will have my newly defined "my_hw3" environment.


You can comment out test and train function calls in the main of the files.

**USAGE**

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python sac.py

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python vanilla_pg.py


## SAC PLOT 
![SAC](/sac_rewards.png)


## VPG PLOT (Different Reward Function)
![VPG](/vpg_rewards.png)


## Details
### Vanilla Policy Gradient (VPG)

- Simple, policy network, On-policy
- Implements the REINFORCE algorithm with a baseline
- Trains for 5000 episodes with a discount factor of 0.99
- Uses Adam optimizer with learning rate of 1e-3

### Soft Actor-Critic (SAC)

- Off-policy reinforcement learning
- Uses a Gaussian policy network and dual Q-networks
- Trains for 2000 episodes with a discount factor of 0.99
- Uses Adam optimizer with learning rate of 3e-4
- Includes target networks with soft updates (tau=0.005)
