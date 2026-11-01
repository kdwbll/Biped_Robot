"""Experimental GPU VecEnv. Must pass parity checks before production use."""
from pathlib import Path
import math,time
from types import SimpleNamespace
import numpy as np,torch,mujoco,warp as wp,mujoco_warp as mjw
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stabilized_servo import StabilizedServo
from warp_servo import WarpServo
from live_telemetry import open_buffer,write_slot

class WarpVecEnv(VecEnv):
 def __init__(self,nworld=32,weight=.5,seconds=12.,seed=42,telemetry_tag=None):
  self.root=Path(__file__).parent;wp.config.kernel_cache_dir=str(self.root/'warp_cache');wp.init()
  self.m=mujoco.MjModel.from_xml_path(str(self.root/'biped_filtered_collision_experimental.xml'))
  self.cpu=mujoco.MjData(self.m);mujoco.mj_resetDataKeyframe(self.m,self.cpu,0);mujoco.mj_forward(self.m,self.cpu)
  initial=StabilizedServo(self.m,self.cpu);self.qpos0=self.cpu.qpos.copy();self.initial=initial.initial_positions.copy()
  self.body=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');self.ground=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
  self.feet=[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('L0_visual6','L1_visual6')]
  self.verts=[]
  for g in self.feet:
   mesh=self.m.geom_dataid[g];i=self.m.mesh_vertadr[mesh];n=self.m.mesh_vertnum[mesh];self.verts.append(self.m.mesh_vert[i:i+n].copy())
  self.weight=float(weight);self.lift_gain=.25;self.horizon=round(seconds/.02);self.rng=np.random.default_rng(seed);self.nworld=nworld;self.render_mode=None
  with wp.ScopedDevice('cuda:0'):
   self.gm=mjw.put_model(self.m);self.gd=mjw.put_data(self.m,self.cpu,nworld=nworld,nconmax=1024,njmax=4096)
   self.servo=WarpServo(self.gm,self.gd,self.initial,nworld)
  self.count=np.zeros(nworld,dtype=int);self.previous=np.zeros((nworld,5));self.last_feet=np.zeros((nworld,2,2))
  self.last_single=np.full(nworld,-1);self.last_switch=np.full(nworld,-100);self.actions=None
  self.telemetry_map=self.telemetry=None;self.telemetry_last=0.
  if telemetry_tag:
   self.telemetry_map,self.telemetry=open_buffer(telemetry_tag);self.telemetry[:]=0.
  super().__init__(nworld,spaces.Box(-np.inf,np.inf,shape=(18,),dtype=np.float32),spaces.Box(-1,1,shape=(5,),dtype=np.float32))

 def _arrays(self):
  wp.synchronize();d=self.gd
  return dict(q=d.qpos.numpy(),v=d.qvel.numpy(),rot=d.xmat.numpy()[:,self.body],irot=d.ximat.numpy()[:,self.body],vel=d.cvel.numpy()[:,self.body,:3],
   body=d.xipos.numpy()[:,self.body],pos=d.geom_xpos.numpy()[:,self.feet],mat=d.geom_xmat.numpy()[:,self.feet],
   targets=self.servo.targets.numpy(),overflow=d.overflow.numpy())
 def _obs(self,a):
  gravity=-a['rot'][:,2,:];omega=np.einsum('nji,nj->ni',a['irot'],a['vel'])
  phase=2*np.pi*self.count*.02
  return np.concatenate((gravity,omega,self.previous,a['targets']-self.initial,np.sin(phase)[:,None],np.cos(phase)[:,None]),axis=1).astype(np.float32)
 def _contacts(self):
  d=self.gd;n=int(d.nacon.numpy()[0]);geom=d.contact.geom.numpy()[:n];world=d.contact.worldid.numpy()[:n]
  out=np.zeros((self.nworld,2),dtype=bool)
  for i,foot in enumerate(self.feet):
   hit=((geom[:,0]==self.ground)&(geom[:,1]==foot))|((geom[:,1]==self.ground)&(geom[:,0]==foot))
   ids=world[hit];ids=ids[(ids>=0)&(ids<self.nworld)];out[ids,i]=True
  return out
 def _low(self,a):
  lows=[]
  for i in range(2):lows.append(np.min(self.verts[i]@a['mat'][:,i,2,:].T,axis=0)+a['pos'][:,i,2])
  return np.stack(lows,axis=1)
 def _reset_mask(self,mask):
  mask=np.asarray(mask,dtype=bool);ids=np.flatnonzero(mask)
  if not len(ids):return
  with wp.ScopedDevice('cuda:0'):
   mjw.reset_data(self.gm,self.gd,wp.array(mask,dtype=wp.bool,device='cuda:0'));wp.synchronize()
   q=wp.to_torch(self.gd.qpos);v=wp.to_torch(self.gd.qvel);targets=wp.to_torch(self.servo.targets)
   idx=torch.as_tensor(ids,device='cuda',dtype=torch.long)
   q[idx]=torch.as_tensor(self.qpos0,device='cuda',dtype=torch.float32)
   v[idx,3:6]=torch.as_tensor(self.rng.uniform(-.02,.02,size=(len(ids),3)),device='cuda',dtype=torch.float32)
   targets[idx]=torch.as_tensor(self.initial,device='cuda',dtype=torch.float32);torch.cuda.synchronize()
   mjw.forward(self.gm,self.gd);wp.synchronize()
  self.count[ids]=0;self.previous[ids]=0;self.last_single[ids]=-1;self.last_switch[ids]=-100
  self.last_feet[ids]=self.gd.geom_xpos.numpy()[ids][:,self.feet,:2]
 def reset(self):
  self._reset_mask(np.ones(self.nworld,dtype=bool));return self._obs(self._arrays())
 def step_async(self,actions):self.actions=np.asarray(actions,dtype=np.float32)
 def step_wait(self):
  if self.actions.shape!=(self.nworld,5) or not np.isfinite(self.actions).all():raise ValueError('invalid actions')
  old=self._arrays();phase=np.sin(2*np.pi*self.count*.02);left=np.maximum(phase,0);right=np.maximum(-phase,0)
  ref=np.stack((-left,left,right,-right,np.zeros_like(left)),axis=1);act=np.clip(ref+.3*self.actions,-1,1)
  with wp.ScopedDevice('cuda:0'):
   self.servo.apply_support(self.body,3.75*self.weight,self.weight*sum(self.m.body_mass)*9.81)
   self.servo.set_actions(act);self.servo.control_step()
  a=self._arrays();self.count+=1;contacts=self._contacts();low=self._low(a);upright=a['rot'][:,2,2];dx=a['body'][:,0]-old['body'][:,0]
  now=time.monotonic()
  if self.telemetry is not None and now-self.telemetry_last>.15:
   for i in range(min(self.nworld,8)):
    write_slot(self.telemetry,i,SimpleNamespace(qpos=a['q'][i],qvel=a['v'][i],time=self.count[i]*.02),self.weight)
   self.telemetry_last=now
  finite=np.isfinite(a['q']).all(axis=1)&np.isfinite(a['v']).all(axis=1)
  base_term=(upright<np.cos(np.deg2rad(45)))|(a['body'][:,2]<.12)
  term=(upright<np.cos(np.deg2rad(35)))|base_term|~finite
  trunc=(self.count>=self.horizon)&~term;done=term|trunc
  vel=np.clip(dx/.02,-.3,.3);reward=.01*np.maximum(0,upright)-.01*np.abs(vel-.05)/.05-.001*np.mean(act**2,axis=1)-.001*np.mean((act-self.previous)**2,axis=1)
  tilt=np.arccos(np.clip(upright,-1,1));reward-=.03*np.minimum((tilt/.35)**2,4);reward-=base_term.astype(float);reward-=2.*(term&~base_term)
  single=np.where(contacts[:,0]&~contacts[:,1],0,np.where(contacts[:,1]&~contacts[:,0],1,-1))
  switched=(single>=0)&(self.last_single>=0)&(single!=self.last_single)&(self.count-self.last_switch>=6)
  reward+=.02*(switched&(dx>0));self.last_switch[switched]=self.count[switched];valid=single>=0;self.last_single[valid]=single[valid]
  swing=((contacts[:,0]&~contacts[:,1]&(low[:,1]>.003))|(contacts[:,1]&~contacts[:,0]&(low[:,0]>.003)));reward+=.003*swing
  phase=np.sin(2*np.pi*self.count*.02);s=np.where(phase>=0,0,1);stance=1-s;idx=np.arange(self.nworld)
  reward+=.025*np.clip(low[idx,s]/.01,0,1)*contacts[idx,stance]
  reward+=np.where(s==0,self.lift_gain,.10)*np.clip(low[idx,s]/.006,0,1)*contacts[idx,stance]
  drift=(np.linalg.norm(a['pos'][:,:,:2]-self.last_feet,axis=2)*contacts).sum(axis=1);reward-=.5*drift
  self.last_feet=a['pos'][:,:,:2].copy();self.previous=act.copy();ob=self._obs(a)
  infos=[dict(foot_contacts=contacts[i].tolist(),foot_clearance_m=low[i].tolist(),overflow=int(a['overflow'][i]),TimeLimit_truncated=bool(trunc[i])) for i in range(self.nworld)]
  if np.any(a['overflow']):raise RuntimeError('MuJoCo Warp capacity overflow; trajectory invalid')
  for i in np.flatnonzero(done):infos[i]['terminal_observation']=ob[i].copy()
  if done.any():self._reset_mask(done);ob[done]=self._obs(self._arrays())[done]
  return ob,reward.astype(np.float32),done,infos
 def close(self):pass
 def get_attr(self,attr_name,indices=None):return [getattr(self,attr_name) for _ in self._get_indices(indices)]
 def set_attr(self,attr_name,value,indices=None):setattr(self,attr_name,value)
 def env_method(self,method_name,*method_args,indices=None,**method_kwargs):return [getattr(self,method_name)(*method_args,**method_kwargs) for _ in self._get_indices(indices)]
 def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]
