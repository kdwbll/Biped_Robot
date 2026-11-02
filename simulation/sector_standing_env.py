"""Constrained simulation-only standing environment for training-readiness tests.

This environment is experimental: collisions and servo parameters are incomplete.
Do not use it for real hardware or claim that a learned policy transfers.
"""
from pathlib import Path
from dataclasses import replace
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec

class SectorStandingEnv(gym.Env):
    metadata={'render_modes': []}
    def __init__(self,episode_seconds=4.0):
        self.model=mujoco.MjModel.from_xml_path(str(Path(__file__).parent/'diagnostic_sector_collision_trial.xml'))
        self.model.opt.timestep=.00025
        if self.model.nu!=5 or self.model.npair!=12851:raise ValueError('Unexpected sector candidate')
        self.data=mujoco.MjData(self.model)
        self.body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
        self.ground=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'ground')
        joints=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate') for i in (0,1)]
        if min(self.body,self.ground,*joints)<0:raise ValueError('Missing body or hip joints')
        self.hip_qadr=[int(self.model.jnt_qposadr[j]) for j in joints]
        self.action_space=spaces.Box(-1.,1.,shape=(5,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(16,),dtype=np.float32)
        if not np.isfinite(episode_seconds) or episode_seconds<=0:raise ValueError('Episode duration must be positive')
        self.control_dt=.02;self.steps_per_control=round(self.control_dt/self.model.opt.timestep)
        if not np.isclose(self.steps_per_control*self.model.opt.timestep,self.control_dt):raise ValueError('Invalid timestep')
        self.horizon=max(1,round(episode_seconds/self.control_dt));self.amplitude_us=np.array([30.,30.,30.,30.,20.]);self.hip_limit_rad=np.deg2rad(4.5)
        self.controller=None;self.previous=np.zeros(5);self.count=0;self.done=True
    def _observation(self):
        R=self.data.xmat[self.body].reshape(3,3);gravity=R.T@np.array([0.,0.,-1.]);velocity=np.empty(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.body,velocity,1)
        targets=self.controller.target_positions-self.controller.initial_positions
        return np.concatenate([gravity,velocity[:3],self.previous,targets]).astype(np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);mujoco.mj_resetDataKeyframe(self.model,self.data,0);mujoco.mj_forward(self.model,self.data)
        self.initial_hips=self.data.qpos[self.hip_qadr].copy();self.initial_height=float(self.data.xipos[self.body,2]);self.previous=np.zeros(5);self.count=0;self.done=False
        self.controller=CandidateStandingServo(self.model,self.data,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
        self.initial_shaft=float(self.data.actuator_length[4])
        return self._observation(),{'scope':'simulation-only standing; assumed servos and incomplete collision'}
    def step(self,action):
        if self.done:raise RuntimeError('Call reset after episode ends')
        action=np.asarray(action,dtype=float)
        if action.shape!=(5,) or not np.isfinite(action).all():raise ValueError('Expected five finite actions')
        action=np.clip(action,-1,1);prev=self.previous.copy();self.controller.set_pulse_us(1500+self.amplitude_us*action);reason=None;peak_contact=0.;peak_pair=None;max_gear_residual=0.
        for substep in range(self.steps_per_control):
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():reason='nonfinite';break
            if np.any(np.abs(self.data.qpos[self.hip_qadr]-self.initial_hips)>self.hip_limit_rad):reason='hip_range';break
            before=float(self.data.time);self.controller.step()
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or self.data.time<before+self.model.opt.timestep*.5:reason='nonfinite_or_reset';break
            if any(self.data.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL)):reason='capacity';break
            if np.any(np.abs(self.data.qpos[self.hip_qadr]-self.initial_hips)>self.hip_limit_rad):reason='hip_range';break
            residual=(self.data.qpos[self.hip_qadr[0]]-self.initial_hips[0])+.5*(self.data.actuator_length[4]-self.initial_shaft)
            max_gear_residual=max(max_gear_residual,abs(float(np.rad2deg(residual))))
            if substep%10==0:
                for ci,contact in enumerate(self.data.contact[:self.data.ncon]):
                    if self.ground in (contact.geom1,contact.geom2):continue
                    force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,ci,force)
                    if force[0]>peak_contact:
                        peak_contact=float(force[0]);peak_pair=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,contact.geom1),mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,contact.geom2)]
        self.previous=action.copy();self.count+=1
        if reason:
            self.done=True;return np.zeros(16,dtype=np.float32),-1.,True,False,{'reason':reason,'simulation_time':float(self.data.time),'sampled_peak_self_contact_n':peak_contact,'sampled_peak_pair':peak_pair,'max_gear_residual_deg':max_gear_residual}
        mujoco.mj_forward(self.model,self.data)
        upright=float(self.data.xmat[self.body].reshape(3,3)[2,2]);height=float(self.data.xipos[self.body,2])
        if upright<np.cos(np.deg2rad(35)) or height<self.initial_height-.08:reason='fall'
        terminated=reason is not None;truncated=self.count>=self.horizon and not terminated;self.done=terminated or truncated
        reward=float(.02*max(0.,upright)-.001*np.mean(action*action)-.001*np.mean((action-prev)**2)-(1. if terminated else 0.))
        return self._observation(),reward,terminated,truncated,{'reason':reason,'upright':upright,'height':height,'simulation_time':float(self.data.time),'max_hip_deg':float(np.max(np.abs(np.rad2deg(self.data.qpos[self.hip_qadr]-self.initial_hips)))),'sampled_peak_self_contact_n':peak_contact,'sampled_peak_pair':peak_pair,'max_gear_residual_deg':max_gear_residual}
