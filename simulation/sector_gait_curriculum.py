"""Experimental phase-conditioned foot-unloading curriculum for simulation only."""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper

class GaitCurriculum(VecEnvWrapper):
    def __init__(self,env,period_s=1.6,objective='unload'):
        if not np.isfinite(period_s) or period_s<=0:raise ValueError('Invalid gait period')
        if objective not in ('unload','load_shift'):raise ValueError('Unknown objective')
        self.objective=objective
        self.period_s=float(period_s);self.phase=np.zeros(env.num_envs,dtype=float)
        low=np.concatenate((env.observation_space.low,np.full(2,-1,dtype=np.float32)))
        high=np.concatenate((env.observation_space.high,np.full(2,1,dtype=np.float32)))
        super().__init__(env,observation_space=spaces.Box(low,high,dtype=np.float32))
    def _augment(self,obs):
        angle=2*np.pi*self.phase/self.period_s
        return np.concatenate((obs,np.sin(angle)[:,None],np.cos(angle)[:,None]),axis=1).astype(np.float32)
    def reset(self):
        obs=self.venv.reset();self.phase[:]=0.;return self._augment(obs)
    def step_wait(self):
        obs,reward,done,infos=self.venv.step_wait();angle=2*np.pi*self.phase/self.period_s
        for i,info in enumerate(infos):
            forces=np.asarray(info['sampled_sole_peak_normal_n']);contact=forces>=.5
            swing=0 if np.sin(angle[i])>=0 else 1;stance=1-swing
            # Heuristic peak-contact signal, not measured ground clearance.
            load=np.asarray(info['sampled_sole_mean_load_n']);total=float(load.sum())
            alignment=float(np.sin(angle[i])*(load[1]-load[0])/total) if total>2. else 0.
            if self.objective=='load_shift':extra=.02*.5*alignment if not info['guard_flags'] else 0.
            else:extra=.02*(.30*float(not contact[swing])+.20*float(contact[stance])) if not info['guard_flags'] else 0.
            reward[i]+=extra;info['curriculum_swing_leg']=swing;info['curriculum_sampled_contact']=contact.tolist();info['curriculum_bonus']=extra
            info['curriculum_load_alignment']=alignment
            if done[i] and 'terminal_observation' in info:
                info['terminal_observation']=np.concatenate((info['terminal_observation'],[np.sin(angle[i]),np.cos(angle[i])])).astype(np.float32)
        self.phase+=.02;self.phase[done]=0.
        return self._augment(obs),reward,done,infos
