"""Reward iteration: discourage leaning and encourage genuine supported swing."""
import numpy as np
from clocked_biped_env import ClockedBipedEnv
import mujoco
class SaferClockedEnv(ClockedBipedEnv):
    def __init__(self,episode_seconds=4.0,disturbance=.02):
        super().__init__(episode_seconds,disturbance)
        self.foot_geom=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('L0_visual6','L1_visual6')]
    def step(self,action):
        obs,reward,term,trunc,info=super().step(action)
        if info.get('reason')=='nonfinite':return obs,reward,term,trunc,info
        tilt=float(np.arccos(np.clip(info['upright'],-1,1)))
        penalty=.03*min((tilt/.35)**2,4.)
        lows=[]
        for g in self.foot_geom:
            mesh=self.model.geom_dataid[g];i=self.model.mesh_vertadr[mesh];n=self.model.mesh_vertnum[mesh]
            verts=self.model.mesh_vert[i:i+n]@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g]
            lows.append(float(np.min(verts[:,2])))
        c=info.get('foot_contacts',[False,False])
        swing=(c[0] and not c[1] and lows[1]>.003) or (c[1] and not c[0] and lows[0]>.003)
        reward=reward-penalty+(.003 if swing else 0.)
        if tilt > np.deg2rad(35):
            if not term: reward -= 2.0
            term=True;trunc=False;info['reason']='tilt_limit_35deg'
        info['foot_clearance_m']=lows;info['valid_swing']=bool(swing)
        return obs,float(reward),term,trunc,info

