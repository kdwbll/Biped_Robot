"""Curriculum: explicit alternating supported foot clearance, simulator only."""
import numpy as np
import mujoco
from safer_clocked_env import SaferClockedEnv
class SteppingEnv(SaferClockedEnv):
    def __init__(self,episode_seconds=4.0,disturbance=.02):
        super().__init__(episode_seconds,disturbance)
        self.last_feet=np.zeros((2,2))
    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        self.last_feet=np.array([self.data.geom_xpos[g,:2] for g in self.foot_geom])
        return obs,info
    def step(self,action):
        obs,reward,term,trunc,info=super().step(action)
        if info.get('reason')=='nonfinite':return obs,reward,term,trunc,info
        feet=info['foot_contacts'];low=info['foot_clearance_m'];pos=np.array([self.data.geom_xpos[g,:2] for g in self.foot_geom]);phase=np.sin(2*np.pi*self.count*.02)
        swing=0 if phase>=0 else 1;stance=1-swing
        # Reward measurable clearance only when opposite foot supports the body.
        lift_reward=.025*np.clip(low[swing]/.01,0,1) if feet[stance] else 0.
        # Approximate support-point drift; visually inspected later to rule out skating.
        drift=sum(np.linalg.norm(pos[i]-self.last_feet[i]) for i in range(2) if feet[i])
        self.last_feet=pos;reward=float(reward+lift_reward-.5*drift)
        info['phase_swing']=swing;info['lift_shaping']=float(lift_reward);info['support_drift_m']=float(drift)
        return obs,reward,term,trunc,info
