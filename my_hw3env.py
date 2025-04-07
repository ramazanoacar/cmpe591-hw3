import torch
import torchvision.transforms as transforms
import numpy as np
import time

import environment

class Hw3Env(environment.BaseEnv):
    def __init__(self, render_mode="offscreen", **kwargs):
        self._render_mode = render_mode
        self.viewer = None
        self._init_position = [-np.pi/2, -np.pi/2, np.pi/2, -2.07, 0, 0, 200]
        self._joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
            "ur5e/robotiq_2f85/right_driver_joint"
        ]
        self._ee_site = "ur5e/robotiq_2f85/gripper_site"
        # Call the BaseEnv reset routine from your environment
        self.reset()
        self._joint_qpos_idxs = [self.model.joint(x).qposadr for x in self._joint_names]
        
        # Additional task parameters
        self._delta = 0.05
        self._goal_thresh = 0.075
        self._max_timesteps = 300
        self._prev_obj_pos = None

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        # Randomize object and goal positions
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def reset(self):
        # Call the base reset routine: create scene, build model, and create viewer.
        scene = self._create_scene()
        xml_string = scene.to_xml_string()
        assets = scene.get_assets()
        self.model = environment.mujoco.MjModel.from_xml_string(xml_string, assets=assets)
        self.data = environment.mujoco.MjData(self.model)
        if self._render_mode == "gui":
            self.viewer = environment.mujoco_viewer.MujocoViewer(self.model, self.data)
        elif self._render_mode == "offscreen":
            self.viewer = environment.mujoco.Renderer(self.model, 128, 128)
        self.data.ctrl[:] = self._init_position
        environment.mujoco.mj_step(self.model, self.data, nstep=2000)
        self.data.ctrl[4] = -np.pi/2
        environment.mujoco.mj_step(self.model, self.data, nstep=2000)
        self._t = 0
        self._prev_obj_pos = self.data.body("obj1").xpos[:2].copy()
        return self.high_level_state()

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        d_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        d_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
        r_ee_to_obj = -0.1 * d_ee_to_obj
        r_obj_to_goal = -0.2 * d_obj_to_goal
        # Direction bonus: reward if object moves toward goal
        obj_movement = obj_pos - self._prev_obj_pos
        dir_to_goal = (goal_pos - obj_pos) / (np.linalg.norm(goal_pos - obj_pos) + 1e-8)
        r_direction = 0.5 * max(0, np.dot(obj_movement/(np.linalg.norm(obj_movement)+1e-8), dir_to_goal))
        if np.linalg.norm(obj_movement) < 1e-6:
            r_direction = 0.0
        r_terminal = 10.0 if self.is_terminal() else 0.0
        r_step = -0.1
        self._prev_obj_pos = obj_pos.copy()
        return r_ee_to_obj + r_obj_to_goal + r_direction + r_terminal + r_step

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action):
        action = action.clamp(-1, 1).cpu().numpy() * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
        result = self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
        self._t += 1
        next_state = self.high_level_state()
        reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated() if result else True
        return next_state, reward, terminal, truncated

