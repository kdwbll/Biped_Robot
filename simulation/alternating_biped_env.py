"""Step-transition shaped reward; experimental simulator only."""
import numpy as np
import mujoco
from randomized_biped_env import PerturbedBipedEnv

class AlternatingBipedEnv(PerturbedBipedEnv):
    def __init__(self, episode_seconds=4.0, disturbance=.02):
        super().__init__(episode_seconds,disturbance)
        self.last_single=None
        self.last_switch_step=-100
        self.switches=0
    def reset(self, *, seed=None, options=None):
        obs,info=super().reset(seed=seed,options=options)
        self.last_single=None;self.last_switch_step=-100;self.switches=0
        return obs,info
    def step(self,action):
        obs,reward,term,trunc,info=super().step(action)
        if info.get('reason')=='nonfinite':return obs,reward,term,trunc,info
        feet=[False,False]
        for c in self.data.contact[:self.data.ncon]:
            a=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or ''
            b=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or ''
            if 'ground' not in (a,b):continue
            x=b if a=='ground' else a
            for i in range(2):
                if x==f'L{i}_visual6':feet[i]=True
        single=0 if feet==[True,False] else (1 if feet==[False,True] else None)
        switched=single is not None and self.last_single is not None and single!=self.last_single and self.count-self.last_switch_step>=6
        if single is not None:
            if switched:
                self.last_switch_step=self.count;self.switches+=1
            self.last_single=single
        # Modest transition reward, gated on forward motion to reduce hopping-in-place exploits.
        reward += .02*float(switched and info['x_displacement']>0.)
        info['foot_contacts']=feet;info['support_switches']=self.switches
        return obs,reward,term,trunc,info
