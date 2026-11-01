"""Learn bounded corrections to a cyclic reference; all virtual support is optional."""
import numpy as np
from supported_stepping_env import SupportedSteppingEnv

class ResidualGaitEnv(SupportedSteppingEnv):
    def __init__(self,episode_seconds=4.,disturbance=.02,support_gain=3.,weight_support=.8):
        super().__init__(episode_seconds,disturbance,support_gain,weight_support)
    def step(self,action):
        phase=np.sin(2*np.pi*self.count*.02)
        left=max(0.,phase);right=max(0.,-phase)
        reference=np.array([-left,left,right,-right,0.])
        applied=np.clip(reference+.3*np.asarray(action),-1.,1.)
        obs,reward,term,trunc,info=super().step(applied)
        info['reference_action']=reference;info['applied_action']=applied
        return obs,reward,term,trunc,info
