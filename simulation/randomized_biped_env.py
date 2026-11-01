"""Adds seeded tiny angular velocity disturbances without moving the initial feet."""
import numpy as np
import mujoco
from biped_env import BipedEnv

class PerturbedBipedEnv(BipedEnv):
    def __init__(self, episode_seconds=4.0, disturbance=0.02):
        super().__init__(episode_seconds=episode_seconds)
        self.disturbance=float(disturbance)
    def reset(self, *, seed=None, options=None):
        obs,info=super().reset(seed=seed,options=options)
        # Free-joint angular velocities: symmetric and reproducibly generated.
        self.data.qvel[3:6]=self.np_random.uniform(-self.disturbance,self.disturbance,size=3)
        mujoco.mj_forward(self.model,self.data)
        info['initial_angular_velocity']=self.data.qvel[3:6].copy()
        return self._observation(),info
