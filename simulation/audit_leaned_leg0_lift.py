"""Probe leg-0 lift from preset body tilts, without virtual support forces."""
from pathlib import Path
import json
import sys
import numpy as np
import warp as wp
import mujoco_warp as mjw
from sector_reset_pose import tilted_pose
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent;command_start=int(sys.argv[1]) if len(sys.argv)>1 else 25;angles=np.repeat([-8.,-4.,0.,4.,8.],4);commands=np.tile(np.array([[-1.,0.],[0.,1.],[-1.,1.],[0.,0.]]),(5,1));n=len(angles)
env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=3.,on_guard='terminate',pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 env.reset();q=env.gd.qpos.numpy()
 for i,angle in enumerate(angles):q[i],_=tilted_pose(env.m,env.initial_q,env.ground,angle,0.)
 wp.copy(env.gd.qpos,wp.array(q,dtype=wp.float32,device='cuda:0'));mjw.forward(env.gm,env.gd);wp.synchronize()
 mesh=int(env.m.geom_dataid[env.soles[0]]);start=int(env.m.mesh_vertadr[mesh]);count=int(env.m.mesh_vertnum[mesh]);verts=env.m.mesh_vert[start:start+count].copy()
 def clearance():
  pos=env.gd.geom_xpos.numpy()[:,env.soles[0]];rot=env.gd.geom_xmat.numpy()[:,env.soles[0]].reshape(n,3,3)
  return (pos[:,2,None]+rot[:,2,:]@verts.T).min(axis=1)
 initial=clearance();peak=np.zeros(n);bestabs=np.zeros(n);count_ok=np.zeros(n,int);guards=np.zeros(n,int);falls=np.zeros(n,int);alive=np.ones(n,bool);last_upright=np.ones(n)
 for step in range(125):
  action=np.zeros((n,5))
  if step>=command_start:action[:,:2]=commands
  _,_,done,infos=env.step(action)
  for i,info in enumerate(infos):
   if done[i] and info['guard_flags'] and alive[i]:guards[i]+=1;alive[i]=False
   elif done[i] and not info['TimeLimit.truncated'] and alive[i]:falls[i]+=1;alive[i]=False
   if alive[i]:last_upright[i]=info['upright']
  z=clearance();support=np.array([x['upright']>np.cos(np.deg2rad(10)) and x['sampled_sole_peak_normal_n'][1]>.5 for x in infos])&alive&(step>=command_start)
  extra=z-initial;peak=np.maximum(peak,np.where(support,extra,0));bestabs=np.maximum(bestabs,np.where(support,z,0));count_ok+=support&(z>.001)&(extra>.001)
 rows=[{'initial_roll_deg':float(angles[i]),'leg0_action':commands[i].tolist(),'initial_clearance_m':float(initial[i]),'peak_extra_clearance_m':float(peak[i]),'peak_absolute_clearance_m':float(bestabs[i]),'supported_lift_frames':int(count_ok[i]),'guard_count':int(guards[i]),'fall_count':int(falls[i]),'final_upright':float(last_upright[i])} for i in range(n)]
 out={'command_start_frame':command_start,'results':rows,'limitations':'Preset free-body tilt is an artificial diagnostic initial condition, not proof it can be reached safely. One pose per combination, sampled at 20 ms. No virtual support forces, learning, walking or hardware validation.'}
 (p/f'leaned_leg0_lift_start_{command_start}_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps([r for r in rows if r['supported_lift_frames']],indent=2),flush=True)
finally:env.close()
