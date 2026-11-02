"""Search bounded perturbations of a candidate for safer supported clearance."""
from pathlib import Path
import json
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent;candidate=json.loads((p/'exploration_clearance_audit.json').read_text())['best'];base=np.asarray(candidate['action_history']);leg=candidate['leg'];n=32
rng=np.random.default_rng(7201);schedules=np.clip(base[None,:,:]+rng.normal(0,.18,(n,len(base),5)),-1,1);schedules[0]=base
env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=2.,on_guard='terminate',reset_tilt_deg=1.,pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 env.seed(7201);env.reset();mesh=int(env.m.geom_dataid[env.soles[leg]]);start=int(env.m.mesh_vertadr[mesh]);count=int(env.m.mesh_vertnum[mesh]);verts=env.m.mesh_vert[start:start+count].copy()
 peak=np.zeros(n);counts=np.zeros(n,int);guards=np.zeros(n,int);falls=np.zeros(n,int);alive=np.ones(n,bool);last_upright=np.ones(n)
 for step in range(90):
  _,_,done,infos=env.step(schedules[:,min(step//30,len(base)-1),:])
  for i,info in enumerate(infos):
   if done[i] and info['guard_flags']:guards[i]+=1;alive[i]=False
   elif done[i] and not info['TimeLimit.truncated']:falls[i]+=1;alive[i]=False
   if alive[i]:last_upright[i]=info['upright']
  pos=env.gd.geom_xpos.numpy()[:,env.soles[leg]];rot=env.gd.geom_xmat.numpy()[:,env.soles[leg]].reshape(n,3,3);low=(pos[:,2,None]+rot[:,2,:]@verts.T).min(axis=1)
  support=np.array([x['upright']>np.cos(np.deg2rad(10)) and x['sampled_sole_peak_normal_n'][1-leg]>.5 for x in infos])&alive
  peak=np.maximum(peak,np.where(support,low,0));counts+=support&(low>.001)
 eligible=alive&(counts>0)&(last_upright>np.cos(np.deg2rad(10)))
 best=None
 if np.any(eligible):
  i=int(np.argmax(np.where(eligible,peak,-1)));best={'world':i,'seed':7201+i,'leg':leg,'peak_clearance_m':float(peak[i]),'supported_frames_over_1mm':int(counts[i]),'final_upright':float(last_upright[i]),'action_history':schedules[i].tolist()}
 out={'best':best,'eligible_worlds':int(eligible.sum()),'guard_events':int(guards.sum()),'falls':int(falls.sum()),
      'limitations':'32 random open-loop candidates, one initial pose each, 1.8 s and 20 ms sampling. Eligible requires no guard and final tilt below 10 degrees. Not trained policy or hardware validation.'}
 (p/'refine_openloop_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
