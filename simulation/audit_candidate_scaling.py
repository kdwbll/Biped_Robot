"""Probe action scaling for supported foot clearance and guard events."""
from pathlib import Path
import json
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent;candidate=json.loads((p/'exploration_clearance_audit.json').read_text())['best'];history=np.asarray(candidate['action_history']);leg=candidate['leg'];scales=np.repeat(np.array([.4,.5,.6,.7,.8,.9,1.,1.1]),4);n=len(scales)
env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=2.,on_guard='terminate',reset_tilt_deg=1.,pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 env.seed(6201);env.reset();mesh=int(env.m.geom_dataid[env.soles[leg]]);start=int(env.m.mesh_vertadr[mesh]);count=int(env.m.mesh_vertnum[mesh]);verts=env.m.mesh_vert[start:start+count].copy()
 peak=np.zeros(n);counts=np.zeros(n,int);guards=np.zeros(n,int);falls=np.zeros(n,int);alive=np.ones(n,bool)
 for step in range(90):
  _,_,done,infos=env.step(scales[:,None]*history[min(step//30,len(history)-1)][None,:])
  for i,info in enumerate(infos):
   if done[i] and info['guard_flags']:guards[i]+=1;alive[i]=False
   elif done[i] and not info['TimeLimit.truncated']:falls[i]+=1;alive[i]=False
  pos=env.gd.geom_xpos.numpy()[:,env.soles[leg]];rot=env.gd.geom_xmat.numpy()[:,env.soles[leg]].reshape(n,3,3)
  low=(pos[:,2,None]+rot[:,2,:]@verts.T).min(axis=1)
  support=np.array([x['upright']>np.cos(np.deg2rad(10)) and x['sampled_sole_peak_normal_n'][1-leg]>.5 for x in infos])&alive
  peak=np.maximum(peak,np.where(support,low,0));counts+=support&(low>.001)
 rows=[]
 for s in np.unique(scales):
  mask=scales==s;rows.append({'scale':float(s),'worlds':int(mask.sum()),'worlds_with_supported_clearance':int(np.sum(counts[mask]>0)),
   'worlds_with_supported_clearance_and_no_guard':int(np.sum((counts[mask]>0)&(guards[mask]==0)&(falls[mask]==0))),
   'guard_events':int(guards[mask].sum()),'fall_events':int(falls[mask].sum()),'peak_clearance_m':float(peak[mask].max())})
 out={'results':rows,'limitations':'Four randomized initial poses per scale, one open-loop sequence, 1.8 seconds, 20 ms sampling; not learned gait, robustness certification or hardware.'}
 (p/'candidate_scaling_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
