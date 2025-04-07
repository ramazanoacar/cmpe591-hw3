# cmpe591-hw3-Ramazan Onur Acar

In this homework, I initially trained SAC and VPG with the previous reward function, the one in the "_homework3.py" because of a misunderstanding.

Later, I did with the newest reward function. However, I faced an issue: Because of some worker issue, my new version of VPG died just before saving the rewards. 

**IMPORTANT**

Therefore, my VPG reward calculations below will have the plot and model from the "_homework3.py"s environment and my SAC will have my newly defined "my_hw3" environment.


You can comment out test and train function calls in the main of the files.

**USAGE**

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python sac.py

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python vanilla_pg.py


## SAC PLOT 
![SAC](/sac_rewards.png)


## VPG PLOT (Different Reward Function)
![VPG](/vpg_rewards.png)
