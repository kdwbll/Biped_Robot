"""Check sole clearance and opposite-foot normal force at the same GPU contact sample."""
from pathlib import Path
import json
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv
p=Path(__file__).parent
histories=[np.asarray(json.loads((p/'训练记录/opposite_lift_20261103_a/best.json').read_text(encoding='utf8'))['action_history']),np.asarray(json.loads((p/'训练记录/lift_search_20261102_a/best.json').read_text(encoding='utf8'))['action_history'])]
n=16;env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=2.,on_guard='terminate',reset_tilt_deg=1.,pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 env.reset_rng=[np.random.default_rng(5123+i%8) for i in range(n)];env.reset();vertices=[]
 for sole in env.soles:
  mesh=int(env.m.geom_dataid[sole]);a=int(env.m.mesh_vertadr[mesh]);vertices.append(env.m.mesh_vert[a:a+int(env.m.mesh_vertnum[mesh])].copy())
 alive=np.ones(n,bool);counts=np.zeros(n,int);peak=np.zeros(n);guards=np.zeros(n,int);upright=np.ones(n);step_counts=np.zeros(n,int);step_peaks=np.zeros(n)
 def sample(e,geom,world,force,guard_done):
  positions=e.gd.geom_xpos.numpy()[:,e.soles]; rotations=e.gd.geom_xmat.numpy()[:,e.soles].reshape(n,2,3,3);rot=e.gd.xmat.numpy()[:,e.body];up=rot[:,2,2]
  for leg in (0,1):
   low=(positions[:,leg,2,None]+rotations[:,leg,2,:]@vertices[leg].T).min(axis=1);opposite=e.soles[1-leg]
   touch=((geom[:,0]==e.ground)&(geom[:,1]==opposite))|((geom[:,1]==e.ground)&(geom[:,0]==opposite));touch &= (world>=0)&(world<n)
   normal=np.zeros(n)
   if np.any(touch):np.maximum.at(normal,world[touch],np.maximum(0,force[touch,0]))
   mask=(np.arange(n)//8==leg)&alive&~guard_done&(up>np.cos(np.deg2rad(10)))&(normal>.5)&(low>.001)
   step_counts[mask]+=1;step_peaks[mask]=np.maximum(step_peaks[mask],low[mask])
 env.diagnostic_sample_observer=sample
 for step in range(90):
  step_counts[:]=0;step_peaks[:]=0;action=np.vstack((np.tile(histories[0][step//30],(8,1)),np.tile(histories[1][step//30],(8,1))))
  _,_,done,infos=env.step(action)
  for i in range(n):
   if alive[i] and done[i]:guards[i]=int(infos[i]['guard_flags']);alive[i]=False
   if alive[i]:upright[i]=infos[i]['upright']
  counts+=np.where(alive,step_counts,0);peak=np.maximum(peak,np.where(alive,step_peaks,0))
 good=alive&(upright>np.cos(np.deg2rad(10)))&(counts>0)
 out={'seeds':'5123 through 5130 shared by each action','worlds_per_leg':8,'synchronized_samples_over_1mm':counts.tolist(),'peak_clearance_m':peak.tolist(),'guard_flags':guards.tolist(),'final_upright':upright.tolist(),'passing_by_leg':[int(good[:8].sum()),int(good[8:].sum())], 'limitations':'Contact force and mesh pose sampled from the same simulation substep every 2.5 ms. A single qualifying sample does not establish continuous support, a step, or reliable hardware behavior. Fixed unmeasured physical model.'}
 (p/'训练记录/synchronized_lift_contact_20261103.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
