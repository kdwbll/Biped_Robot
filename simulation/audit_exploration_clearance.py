"""Search short random open-loop actions for sole clearance; no policy training."""
from pathlib import Path
import json
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent;n=32;env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=4.,on_guard='terminate',reset_tilt_deg=1.,reward_mode='shaped',pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 env.seed(4012);env.reset();rng=np.random.default_rng(4012);verts=[]
 for geom in env.soles:
  mesh=int(env.m.geom_dataid[geom]);start=int(env.m.mesh_vertadr[mesh]);count=int(env.m.mesh_vertnum[mesh]);verts.append(env.m.mesh_vert[start:start+count].copy())
 maxz=np.full((n,2),-1.);counts=np.zeros((n,2),int);guards=np.zeros(n,int);falls=np.zeros(n,int);alive=np.ones(n,bool);actions=[];best={};world_best=[None]*n
 for step in range(90):
  if step%30==0:action=rng.uniform(-1,1,(n,5));action[:,4]*=.5;actions.append(action.copy())
  _,_,done,infos=env.step(action)
  for i,info in enumerate(infos):
   if done[i] and info['guard_flags']:guards[i]+=1;alive[i]=False
   elif done[i] and not info['TimeLimit.truncated']:falls[i]+=1;alive[i]=False
  if step%5==0:
   positions=env.gd.geom_xpos.numpy()[:,env.soles];rot=env.gd.geom_xmat.numpy()[:,env.soles].reshape(n,2,3,3)
   for leg in (0,1):
    z=positions[:,leg,2,None]+rot[:,leg,2,:]@verts[leg].T;low=z.min(axis=1);maxz[:,leg]=np.maximum(maxz[:,leg],np.where(alive,low,-1));counts[:,leg]+=(low>.001)&alive
    supported=np.array([info['upright']>np.cos(np.deg2rad(10)) and info['sampled_sole_peak_normal_n'][1-leg]>.5 for info in infos])
    eligible=alive&supported
    for i in np.flatnonzero(eligible):
     if world_best[i] is None or float(low[i])>world_best[i]['clearance_m']:world_best[i]={'world':int(i),'leg':leg,'step':step,'clearance_m':float(low[i]),'upright':infos[i]['upright'],'opposite_foot_peak_n':infos[i]['sampled_sole_peak_normal_n'][1-leg],'action_history':[x[i].tolist() for x in actions]}
 survivors=[x for i,x in enumerate(world_best) if x is not None and guards[i]==0 and falls[i]==0]
 if survivors:best=max(survivors,key=lambda x:x['clearance_m'])
 out={'seed':4012,'worlds':n,'frames':90,'best':best,'max_clearance_m':maxz.tolist(),'frames_sampled_over_1mm':counts.tolist(),'guard_count_per_world':guards.tolist(),
      'guard_episodes':int(guards.sum()),'fall_episodes':int(falls.sum()),
      'limitations':'One random action schedule, sampled every 0.1 s; best candidate additionally requires tilt below 10 degrees, opposite-foot sampled contact, and no guard/fall through 1.8 s. Fixed assumed mechanics; no training or hardware validation. No claim of a stable step.'}
 (p/'exploration_clearance_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({'best':best,'worlds_with_1mm_clearance':int(np.sum(np.any(counts>0,axis=1))),'guards':int(guards.sum()),'falls':int(falls.sum())},indent=2),flush=True)
finally:env.close()
