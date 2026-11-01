"""Experimental simulation environment; not validated for the physical robot."""
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from theoretical_servo import TheoreticalServoController

class BipedEnv(gym.Env):
    metadata={'render_modes': []}
    def __init__(self, episode_seconds=8.0, model_path='biped_filtered_collision_experimental.xml'):
        self.model=mujoco.MjModel.from_xml_path(str(Path(__file__).parent/model_path))
        self.data=mujoco.MjData(self.model)
        self.body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
        self.action_space=spaces.Box(-1.0,1.0,shape=(5,),dtype=np.float32)
        # gravity direction in body frame, local angular velocity, previous action,
        # and previously commanded targets (all potentially available from IMU/commands).
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(16,),dtype=np.float32)
        self.horizon=int(round(episode_seconds/.02)); self.count=0
        self.action_amplitude_us=80.0; self.previous=np.zeros(5,dtype=np.float64)
        self.controller=None
        self.steps_per_control=round(.02/self.model.opt.timestep)
        if not np.isclose(self.steps_per_control*self.model.opt.timestep,.02): raise ValueError('Bad simulation timestep')

    def _observation(self):
        R=self.data.xmat[self.body].reshape(3,3)
        gravity=R.T@np.array([0.,0.,-1.])
        vel=np.empty(6); mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.body,vel,1)
        targets=self.controller.target_positions-self.controller.initial_positions
        return np.concatenate([gravity,vel[:3],self.previous,targets]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        mujoco.mj_forward(self.model,self.data)
        self.count=0; self.previous.fill(0.)
        self.controller=TheoreticalServoController(self.model,self.data)
        return self._observation(), {'assumptions':'simulation only; ideal servo and approximate collision'}

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all(): raise ValueError('Action must be five finite values')
        action=np.clip(action,-1.,1.); before=self.data.xipos[self.body].copy(); prev=self.previous.copy()
        self.controller.set_pulse_us(1500.+self.action_amplitude_us*action)
        for _ in range(self.steps_per_control): self.controller.step()
        self.previous=action.copy();self.count+=1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return np.zeros(16,dtype=np.float32),-10.,True,False,{'reason':'nonfinite'}
        R=self.data.xmat[self.body].reshape(3,3); upright=float(R[2,2]); height=float(self.data.xipos[self.body,2]); dx=float(self.data.xipos[self.body,0]-before[0]);
        terminated=bool(upright < np.cos(np.deg2rad(45)) or height < .12)
        truncated=bool(self.count>=self.horizon and not terminated)
        velocity=float(np.clip(dx/.02,-.3,.3)); reward=float(.01*max(0.,upright) - .01*abs(velocity-.05)/.05 - .001*np.mean(action**2) - .001*np.mean((action-prev)**2) - (1.0 if terminated else 0.0))
        return self._observation(),reward,terminated,truncated,{'x_displacement':dx,'upright':upright,'height':height}




