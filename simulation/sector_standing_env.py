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
from sector_standing_reward import shaped_standing_reward

class SectorStandingEnv(gym.Env):
    metadata={'render_modes': []}
    def __init__(self,episode_seconds=4.0,max_sampled_self_contact_n=None,model_variant='baseline',max_gear_residual_deg=None,friction_range=None,max_pwm_delta_us_per_control=None,reward_mode='legacy'):
        if reward_mode not in ('legacy','shaped'):raise ValueError('Unknown reward mode')
        self.reward_mode=reward_mode
        variants={'baseline':('diagnostic_sector_collision_trial.xml',12851,.01),'implicit':('diagnostic_sector_implicit_damping_trial.xml',12850,0.),'implicit_stiff':('diagnostic_sector_implicit_stiff_trial.xml',12850,0.)}
        if model_variant not in variants:raise ValueError('Unknown diagnostic model variant')
        filename,expected_pairs,self.hip_damping=variants[model_variant];self.model_variant=model_variant
        self.model=mujoco.MjModel.from_xml_path(str(Path(__file__).parent/filename))
        self.model.opt.timestep=.00025
        if self.model.nu!=5 or self.model.npair!=expected_pairs:raise ValueError('Unexpected sector candidate')
        self.data=mujoco.MjData(self.model)
        self.body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
        self.ground=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'ground')
        joints=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate') for i in (0,1)]
        if min(self.body,self.ground,*joints)<0:raise ValueError('Missing body or hip joints')
        self.hip_qadr=[int(self.model.jnt_qposadr[j]) for j in joints]
        self.hip_dofadr=[int(self.model.jnt_dofadr[j]) for j in joints]
        self.friction_range=None
        if friction_range is not None:
            limits=np.asarray(friction_range,dtype=float)
            if model_variant=='baseline' or limits.shape!=(2,) or not np.isfinite(limits).all() or limits[0]<0 or limits[1]<limits[0]:raise ValueError('Invalid friction range for selected model')
            self.friction_range=limits
        shaft=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')
        if shaft<0 or self.model.actuator_trnid[4,0]!=shaft or not np.allclose(self.model.actuator_gear[4],[1,0,0,0,0,0]):raise ValueError('Unexpected shaft actuator')
        self.shaft_qadr=int(self.model.jnt_qposadr[shaft])
        self.action_space=spaces.Box(-1.,1.,shape=(5,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(16,),dtype=np.float32)
        if not np.isfinite(episode_seconds) or episode_seconds<=0:raise ValueError('Episode duration must be positive')
        self.control_dt=.02;self.steps_per_control=round(self.control_dt/self.model.opt.timestep)
        if not np.isclose(self.steps_per_control*self.model.opt.timestep,self.control_dt):raise ValueError('Invalid timestep')
        if max_sampled_self_contact_n is not None and (not np.isfinite(max_sampled_self_contact_n) or max_sampled_self_contact_n<=0):raise ValueError('Contact guard must be positive or None')
        self.max_sampled_self_contact_n=max_sampled_self_contact_n
        if max_gear_residual_deg is not None and (not np.isfinite(max_gear_residual_deg) or max_gear_residual_deg<=0):raise ValueError('Gear guard must be positive or None')
        self.max_gear_residual_deg=max_gear_residual_deg
        if max_pwm_delta_us_per_control is not None and (not np.isfinite(max_pwm_delta_us_per_control) or max_pwm_delta_us_per_control<=0):raise ValueError('PWM slew limit must be positive or None')
        self.max_pwm_delta_us_per_control=max_pwm_delta_us_per_control
        self.horizon=max(1,round(episode_seconds/self.control_dt));self.amplitude_us=np.array([30.,30.,30.,30.,20.]);self.hip_limit_rad=np.deg2rad(4.5)
        self.controller=None;self.previous=np.zeros(5);self.count=0;self.done=True
    def _observation(self):
        R=self.data.xmat[self.body].reshape(3,3);gravity=R.T@np.array([0.,0.,-1.]);velocity=np.empty(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.body,velocity,1)
        targets=self.controller.target_positions-self.controller.initial_positions
        return np.concatenate([gravity,velocity[:3],self.previous,targets]).astype(np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if self.friction_range is not None:self.model.dof_frictionloss[self.hip_dofadr]=self.np_random.uniform(self.friction_range[0],self.friction_range[1],2)
        mujoco.mj_resetDataKeyframe(self.model,self.data,0);mujoco.mj_forward(self.model,self.data)
        self.applied_pwm_us=np.full(5,1500.);self.initial_hips=self.data.qpos[self.hip_qadr].copy();self.initial_height=float(self.data.xipos[self.body,2]);self.previous=np.zeros(5);self.count=0;self.done=False
        self.controller=CandidateStandingServo(self.model,self.data,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=self.hip_damping)
        self.initial_shaft=float(self.data.qpos[self.shaft_qadr])
        return self._observation(),{'scope':'simulation-only standing; assumed servos and incomplete collision','model_variant':self.model_variant,'hypothetical_hip_friction_nm':self.model.dof_frictionloss[self.hip_dofadr].tolist()}
    def step(self,action):
        if self.done:raise RuntimeError('Call reset after episode ends')
        action=np.asarray(action,dtype=float)
        if action.shape!=(5,) or not np.isfinite(action).all():raise ValueError('Expected five finite actions')
        action=np.clip(action,-1,1);prev=self.previous.copy();desired_pwm=1500+self.amplitude_us*action
        if self.max_pwm_delta_us_per_control is None:self.applied_pwm_us=desired_pwm
        else:self.applied_pwm_us+=np.clip(desired_pwm-self.applied_pwm_us,-self.max_pwm_delta_us_per_control,self.max_pwm_delta_us_per_control)
        self.controller.set_pulse_us(self.applied_pwm_us);reason=None;peak_contact=0.;peak_pair=None;max_gear_residual=0.
        for substep in range(self.steps_per_control):
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():reason='nonfinite';break
            if np.any(np.abs(self.data.qpos[self.hip_qadr]-self.initial_hips)>self.hip_limit_rad):reason='hip_range';break
            residual_deg=abs(float(np.rad2deg((self.data.qpos[self.hip_qadr[0]]-self.initial_hips[0])+.5*(self.data.qpos[self.shaft_qadr]-self.initial_shaft))))
            max_gear_residual=max(max_gear_residual,residual_deg)
            if self.max_gear_residual_deg is not None and residual_deg>self.max_gear_residual_deg:reason='gear_guard';break
            before=float(self.data.time);self.controller.step()
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or self.data.time<before+self.model.opt.timestep*.5:reason='nonfinite_or_reset';break
            if any(self.data.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL)):reason='capacity';break
            if np.any(np.abs(self.data.qpos[self.hip_qadr]-self.initial_hips)>self.hip_limit_rad):reason='hip_range';break
            residual=(self.data.qpos[self.hip_qadr[0]]-self.initial_hips[0])+.5*(self.data.qpos[self.shaft_qadr]-self.initial_shaft)
            max_gear_residual=max(max_gear_residual,abs(float(np.rad2deg(residual))))
            if self.max_gear_residual_deg is not None and max_gear_residual>self.max_gear_residual_deg:reason='gear_guard';break
            if substep%10==0:
                for ci,contact in enumerate(self.data.contact[:self.data.ncon]):
                    if self.ground in (contact.geom1,contact.geom2):continue
                    force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,ci,force)
                    if force[0]>peak_contact:
                        peak_contact=float(force[0]);peak_pair=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,contact.geom1),mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,contact.geom2)]
                if self.max_sampled_self_contact_n is not None and peak_contact>self.max_sampled_self_contact_n:
                    reason='contact_guard';break
        self.previous=action.copy();self.count+=1
        if reason:
            self.done=True;return np.zeros(16,dtype=np.float32),-1.,True,False,{'reason':reason,'simulation_time':float(self.data.time),'sampled_peak_self_contact_n':peak_contact,'sampled_peak_pair':peak_pair,'max_gear_residual_deg':max_gear_residual,'applied_pwm_us':self.applied_pwm_us.tolist()}
        mujoco.mj_forward(self.model,self.data)
        upright=float(self.data.xmat[self.body].reshape(3,3)[2,2]);height=float(self.data.xipos[self.body,2])
        if upright<np.cos(np.deg2rad(35)) or height<self.initial_height-.08:reason='fall'
        terminated=reason is not None;truncated=self.count>=self.horizon and not terminated;self.done=terminated or truncated
        obs=self._observation()
        if self.reward_mode=='shaped':reward=float(shaped_standing_reward(upright,height,self.initial_height,obs[3:6],action,prev,terminated))
        else:reward=float(.02*max(0.,upright)-.001*np.mean(action*action)-.001*np.mean((action-prev)**2)-(1. if terminated else 0.))
        return obs,reward,terminated,truncated,{'reason':reason,'upright':upright,'height':height,'simulation_time':float(self.data.time),'max_hip_deg':float(np.max(np.abs(np.rad2deg(self.data.qpos[self.hip_qadr]-self.initial_hips)))),'sampled_peak_self_contact_n':peak_contact,'sampled_peak_pair':peak_pair,'max_gear_residual_deg':max_gear_residual,'applied_pwm_us':self.applied_pwm_us.tolist()}
