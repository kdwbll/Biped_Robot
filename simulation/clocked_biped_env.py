"""Clock input gives a memoryless policy a gait phase; simulation only."""
import numpy as np
from gymnasium import spaces
from larger_action_env import LargerActionBipedEnv
class ClockedBipedEnv(LargerActionBipedEnv):
    def __init__(self,episode_seconds=4.0,disturbance=.02):
        super().__init__(episode_seconds,disturbance)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(18,),dtype=np.float32)
    def _observation(self):
        angle=2*np.pi*(self.count*.02)/1.0
        return np.r_[super()._observation(),np.sin(angle),np.cos(angle)].astype(np.float32)
