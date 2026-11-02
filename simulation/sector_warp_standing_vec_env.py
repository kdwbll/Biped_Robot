"""Experimental guarded MuJoCo-Warp standing VecEnv; simulation only.

Defaults to aborting the batch on a guard. Experimental terminate mode resets
guarded worlds; any overflow aborts the batch and invalidates the rollout. GPU friction is
shared and fixed, and simulator/CAD agreement is incomplete.
"""
from pathlib import Path
from dataclasses import replace
import sys
import numpy as np
import mujoco
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec

p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import warp as wp
import mujoco_warp as mjw
from sector_warp_guard import scan_state,scan_contacts,restore_masked_state,set_masked_qpos
from sector_warp_servo_kernels import servo_step,update_pwm_targets
from sector_reset_pose import tilted_pose
from sector_standing_reward import shaped_standing_reward

class SectorWarpStandingVecEnv(VecEnv):
 def __init__(self,nworld=4,episode_seconds=4.,contact_limit_n=20.,gear_limit_deg=.2,pwm_slew_us=1.,on_guard='abort',reset_tilt_deg=0.,reward_mode='legacy'):
  if not isinstance(nworld,int) or not 1<=nworld<=64:raise ValueError('Use 1 to 64 diagnostic worlds')
  if not np.isfinite(episode_seconds) or episode_seconds<=0:raise ValueError('Invalid episode duration')
  for v in (contact_limit_n,gear_limit_deg,pwm_slew_us):
   if not np.isfinite(v) or v<=0:raise ValueError('Diagnostic limits must be positive finite')
  if on_guard not in ('abort','terminate'):raise ValueError('Unknown guard policy')
  if reward_mode not in ('legacy','shaped'):raise ValueError('Unknown reward mode')
  self.reward_mode=reward_mode
  if not np.isfinite(reset_tilt_deg) or not 0<=reset_tilt_deg<=2:raise ValueError('Reset tilt must be between 0 and 2 degrees')
  self.reset_tilt_deg=float(reset_tilt_deg);self.reset_rng=[np.random.default_rng() for _ in range(nworld)];self.reset_tilts=np.zeros((nworld,2));self.reset_lifts=np.zeros(nworld)
  self.on_guard=on_guard
  self.nworld=nworld;self.contact_limit=float(contact_limit_n);self.gear_limit=float(gear_limit_deg);self.slew=float(pwm_slew_us);self.horizon=max(1,round(episode_seconds/.02));self.render_mode=None
  wp.config.kernel_cache_dir=str(p/'warp_cache');wp.init();self.m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_stiff_trial.xml'));self.m.opt.timestep=.00025
  if self.m.nu!=5 or self.m.npair!=12850:raise ValueError('Unexpected diagnostic model')
  self.c=mujoco.MjData(self.m);mujoco.mj_resetDataKeyframe(self.m,self.c,0);mujoco.mj_forward(self.m,self.c)
  servo=CandidateStandingServo(self.m,self.c,replace(ServoSpec(),kp_nm_per_rad=4),ki=1.5,kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=0.)
  self.initial_q=self.c.qpos.astype(np.float32).copy();self.initial_target=servo.initial_positions.astype(np.float32).copy()
  self.body=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');self.ground=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
  joint_ids=[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate') for i in (0,1)];shaft_id=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')
  root_id=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,'floating_base')
  if root_id<0 or self.m.jnt_type[root_id]!=mujoco.mjtJoint.mjJNT_FREE or self.m.jnt_qposadr[root_id]!=0:raise ValueError('Expected floating base quaternion at qpos[3:7]')
  if min(self.body,self.ground,shaft_id,*joint_ids)<0 or self.m.actuator_trnid[4,0]!=shaft_id or not np.allclose(self.m.actuator_gear[4],[1,0,0,0,0,0]):raise ValueError('Missing or unexpected diagnostic joints')
  self.hips=[int(self.m.jnt_qposadr[j]) for j in joint_ids];self.shaft=int(self.m.jnt_qposadr[shaft_id])
  with wp.ScopedDevice('cuda:0'):
   self.gm=mjw.put_model(self.m);self.gd=mjw.put_data(self.m,self.c,nworld=nworld,nconmax=4096,njmax=8192);mjw.forward(self.gm,self.gd);wp.synchronize()
   self.initial_q_dev=wp.array(self.initial_q,dtype=wp.float32,device='cuda:0');self.initial_target_dev=wp.array(self.initial_target,dtype=wp.float32,device='cuda:0')
   init=np.tile(self.initial_target,(nworld,1));self.initial=wp.array(init,dtype=wp.float32,device='cuda:0');self.targets=wp.array(init,dtype=wp.float32,device='cuda:0');self.integral=wp.zeros((nworld,5),dtype=wp.float32,device='cuda:0');self.flags=wp.zeros(nworld,dtype=wp.int32,device='cuda:0')
   with wp.ScopedCapture() as cap:
    wp.launch(servo_step,dim=(nworld,5),inputs=[self.targets,self.integral,self.gd.actuator_length,self.gd.actuator_velocity,self.gd.ctrl,self.m.opt.timestep,0.]);mjw.step(self.gm,self.gd)
   self.graph=cap.graph
  self.hip0=self.initial_q[self.hips].copy();self.shaft0=float(self.initial_q[self.shaft]);self.guard_args=[self.gd.qpos,self.gd.qvel,self.flags,self.m.nq,self.m.nv,self.hips[0],self.hips[1],self.shaft,float(self.hip0[0]),float(self.hip0[1]),self.shaft0,float(np.deg2rad(4.5)),float(np.deg2rad(self.gear_limit))]
  self.previous=np.zeros((nworld,5));self.pwm=np.full((nworld,5),1500.);self.count=np.zeros(nworld,dtype=int);self.pending=None;self.aborted=False;self.initial_height=self.gd.xipos.numpy()[:,self.body,2].copy()
  super().__init__(nworld,spaces.Box(-np.inf,np.inf,shape=(16,),dtype=np.float32),spaces.Box(-1.,1.,shape=(5,),dtype=np.float32))
 def _observation(self):
  rot=self.gd.xmat.numpy()[:,self.body];irot=self.gd.ximat.numpy()[:,self.body];vel=self.gd.cvel.numpy()[:,self.body,:3];gravity=-rot[:,2,:];omega=np.einsum('nji,nj->ni',irot,vel)
  target=self.targets.numpy()-self.initial_target[None,:]
  return np.concatenate((gravity,omega,self.previous,target),axis=1).astype(np.float32)
 def _reset_mask(self,mask,refresh=True):
  mask=np.asarray(mask,dtype=bool)
  if mask.shape!=(self.nworld,):raise ValueError('Wrong reset mask')
  if not np.any(mask):return
  with wp.ScopedDevice('cuda:0'):
   wm=wp.array(mask,dtype=wp.bool,device='cuda:0');mjw.reset_data(self.gm,self.gd,wm)
   wp.launch(restore_masked_state,dim=self.nworld,inputs=[self.gd.qpos,self.gd.qvel,self.gd.ctrl,self.targets,self.integral,self.flags,wm,self.initial_q_dev,self.initial_target_dev,self.m.nq,self.m.nv,self.m.nu])
   if refresh and self.reset_tilt_deg:
    poses=np.tile(self.initial_q,(self.nworld,1))
    for i in np.flatnonzero(mask):
     self.reset_tilts[i]=self.reset_rng[i].uniform(-self.reset_tilt_deg,self.reset_tilt_deg,2)
     poses[i],self.reset_lifts[i]=tilted_pose(self.m,self.initial_q,self.ground,*self.reset_tilts[i])
    wp.launch(set_masked_qpos,dim=self.nworld,inputs=[self.gd.qpos,wp.array(poses,dtype=wp.float32,device='cuda:0'),wm,self.m.nq])
   if refresh:mjw.forward(self.gm,self.gd)
   wp.synchronize()
  self.previous[mask]=0.;self.pwm[mask]=1500.;self.count[mask]=0
 def reset(self):
  for i,seed in enumerate(self._seeds):
   if seed is not None:self.reset_rng[i]=np.random.default_rng(seed)
  self._reset_seeds()
  self._reset_mask(np.ones(self.nworld,dtype=bool));self.aborted=False;self.pending=None
  self.reset_infos=[{'scope':'experimental GPU simulation; fixed unmeasured friction and incomplete contacts','tilt_deg':self.reset_tilts[i].tolist(),'ground_lift_m':float(self.reset_lifts[i])} for i in range(self.nworld)]
  return self._observation()
 def step_async(self,actions):
  a=np.asarray(actions,dtype=float)
  if self.aborted:raise RuntimeError('Batch previously aborted; reset required')
  if self.pending is not None:raise RuntimeError('A step is already pending')
  if a.shape!=(self.nworld,5) or not np.isfinite(a).all():raise ValueError('Expected finite world-by-five actions')
  self.pending=np.clip(a,-1,1)
 def step_wait(self):
  if self.pending is None:raise RuntimeError('No pending action')
  action=self.pending;self.pending=None;prev=self.previous.copy();desired=1500.+action*np.array([30.,30.,30.,30.,20.]);self.pwm+=np.clip(desired-self.pwm,-self.slew,self.slew)
  self.previous=action.copy();peak=np.zeros(self.nworld);peak_pair=[None]*self.nworld;guard_done=np.zeros(self.nworld,dtype=bool);guard_codes=np.zeros(self.nworld,dtype=np.int32);terminal=[None]*self.nworld;previous_time=self.gd.time.numpy().copy()
  def check_flags(step):
   wp.launch(scan_state,dim=self.nworld,inputs=self.guard_args);wp.synchronize();flags=self.flags.numpy();overflow=self.gd.overflow.numpy()
   if np.any(overflow):
    self.aborted=True;raise RuntimeError(f'GPU diagnostic overflow: {overflow.tolist()}; discard rollout')
   new=(flags!=0)&~guard_done
   if np.any(new):
    if self.on_guard=='abort':
     self.aborted=True;raise RuntimeError(f'GPU diagnostic batch aborted: {flags.tolist()} at substep {step}; discard rollout')
    current=self._observation()
    for i in np.flatnonzero(new):
     terminal[i]=current[i].copy() if np.isfinite(current[i]).all() and not (flags[i]&1) else np.zeros(16,dtype=np.float32)
    guard_codes[new]=flags[new];guard_done[new]=True
  with wp.ScopedDevice('cuda:0'):
   check_flags(-1)
   if np.any(guard_done):self._reset_mask(guard_done,refresh=False)
   pwm=wp.array(self.pwm.astype(np.float32),dtype=wp.float32,device='cuda:0');wp.launch(update_pwm_targets,dim=(self.nworld,5),inputs=[self.initial,self.targets,pwm])
   for step in range(80):
    check_flags(step)
    if np.any(guard_done):self._reset_mask(guard_done,refresh=False)
    now=self.gd.time.numpy();active=~guard_done
    if step and np.any(now[active]<previous_time[active]+.5*self.m.opt.timestep):self.aborted=True;raise RuntimeError('GPU time reset; discard rollout')
    previous_time=now.copy();wp.capture_launch(self.graph)
    if step%10==0:
     wp.synchronize();n=int(self.gd.nacon.numpy()[0])
     if n:
      ids=wp.array(np.arange(n,dtype=np.int32),dtype=wp.int32,device='cuda:0');force=wp.zeros(n,dtype=wp.spatial_vector,device='cuda:0');mjw.contact_force(self.gm,self.gd,ids,False,force)
      wp.launch(scan_contacts,dim=n,inputs=[self.gd.contact.geom,self.gd.contact.worldid,force,self.flags,n,self.nworld,self.ground,self.contact_limit]);wp.synchronize()
      f=force.numpy();geom=self.gd.contact.geom.numpy()[:n];world=self.gd.contact.worldid.numpy()[:n];valid=(geom[:,0]!=self.ground)&(geom[:,1]!=self.ground)&(world>=0)&(world<self.nworld)
      for idx in np.flatnonzero(valid):
       w=int(world[idx]);normal=float(f[idx,0])
       if normal>peak[w]:
        peak[w]=normal;peak_pair[w]=[mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in geom[idx]]
   check_flags(80)
   end=self.gd.time.numpy();active=~guard_done
   if np.any(end[active]<previous_time[active]+.5*self.m.opt.timestep):self.aborted=True;raise RuntimeError('GPU time reset; discard rollout')
   if np.any(guard_done):self._reset_mask(guard_done,refresh=False)
   mjw.forward(self.gm,self.gd);wp.synchronize()
  self.previous[~guard_done]=action[~guard_done];self.count[~guard_done]+=1;rot=self.gd.xmat.numpy()[:,self.body];height=self.gd.xipos.numpy()[:,self.body,2];upright=rot[:,2,2]
  term=guard_done|(upright<np.cos(np.deg2rad(35)))|(height<self.initial_height-.08);trunc=(self.count>=self.horizon)&~term;done=term|trunc
  obs=self._observation()
  if self.reward_mode=='shaped':reward=shaped_standing_reward(upright,height,self.initial_height,obs[:,3:6],action,prev,term);reward[guard_done]=-1.
  else:reward=.02*np.maximum(0,upright)-.001*np.mean(action**2,axis=1)-.001*np.mean((action-prev)**2,axis=1)-term.astype(float)
  infos=[{'sampled_peak_self_contact_n':float(peak[i]),'sampled_peak_self_contact_pair':peak_pair[i],'upright':float(upright[i]),'height':float(height[i]),'TimeLimit.truncated':bool(trunc[i]),'guard_flags':int(guard_codes[i]),'model_scope':'experimental simulation only'} for i in range(self.nworld)]
  for i in np.flatnonzero(done):infos[i]['terminal_observation']=terminal[i] if guard_done[i] else obs[i].copy()
  if np.any(done):
   self._reset_mask(done);obs=self._observation()
   for i in np.flatnonzero(done):self.reset_infos[i]={'scope':'experimental GPU simulation; fixed unmeasured friction and incomplete contacts','tilt_deg':self.reset_tilts[i].tolist(),'ground_lift_m':float(self.reset_lifts[i])}
  return obs,reward.astype(np.float32),done,infos
 def close(self):wp.synchronize()
 def get_attr(self,attr_name,indices=None):return [getattr(self,attr_name) for _ in self._get_indices(indices)]
 def set_attr(self,attr_name,value,indices=None):
  if len(self._get_indices(indices))!=self.nworld:raise NotImplementedError('Per-world attribute changes unsupported')
  setattr(self,attr_name,value)
 def env_method(self,method_name,*method_args,indices=None,**method_kwargs):
  if len(self._get_indices(indices))!=self.nworld:raise NotImplementedError('Per-world calls unsupported')
  result=getattr(self,method_name)(*method_args,**method_kwargs)
  return [result for _ in self._get_indices(indices)]
 def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]
