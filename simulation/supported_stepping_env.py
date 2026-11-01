"""Training-only, temporary virtual balance torque, always evaluate without it."""
import numpy as np
import mujoco
from stepping_env import SteppingEnv
class SupportedSteppingEnv(SteppingEnv):
    def __init__(self,episode_seconds=4.,disturbance=.02,support_gain=2.0,weight_support=0.0):
        super().__init__(episode_seconds,disturbance);self.support_gain=float(support_gain)
        self.weight_support=float(np.clip(weight_support,0,1))
    def set_support(self,value):self.support_gain=float(np.clip(value,0,3))
    def set_weight_support(self,value):self.weight_support=float(np.clip(value,0,1))
    def step(self,action):
        R=self.data.xmat[self.body].reshape(3,3);up=R[:,2];rotvel=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.body,rotvel,0)
        torque=self.support_gain*np.cross(up,[0.,0.,1.])-.18*self.support_gain*np.array([rotvel[0],rotvel[1],0.])
        self.data.xfrc_applied[self.body,:]=0.;self.data.xfrc_applied[self.body,3:6]=np.clip(torque,-1.5,1.5)
        self.data.xfrc_applied[self.body,2]=self.weight_support*float(np.sum(self.model.body_mass))*9.81
        obs,reward,term,trunc,info=super().step(action);info['virtual_support_gain']=self.support_gain
        return obs,reward,term,trunc,info
