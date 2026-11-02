"""Test an open-loop clearance candidate across 16 randomized initial poses."""
from pathlib import Path
import json,sys
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent;source=json.loads((p/'exploration_clearance_audit.json').read_text())['best'];candidate=json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv)>1 else source;history=np.asarray(candidate['action_history']);leg=source['leg'];n=16
env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=2.,on_guard='terminate',reset_tilt_deg=1.,reward_mode='shaped',pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 env.seed(5123);env.reset();verts=[]
 for geom in env.soles:
  mesh=int(env.m.geom_dataid[geom]);start=int(env.m.mesh_vertadr[mesh]);count=int(env.m.mesh_vertnum[mesh]);verts.append(env.m.mesh_vert[start:start+count].copy())
 best=np.zeros(n);counts=np.zeros(n,int);guards=np.zeros(n,int);falls=np.zeros(n,int);alive=np.ones(n,bool);last_upright=np.ones(n)
 frames=min(candidate.get('step',len(history)*30-1)+15,len(history)*30)
 for step in range(frames):
  _,_,done,infos=env.step(np.tile(history[step//30],(n,1)))
  for i,info in enumerate(infos):
   if done[i] and info['guard_flags']:guards[i]+=1;alive[i]=False
   elif done[i] and not info['TimeLimit.truncated']:falls[i]+=1;alive[i]=False
   if alive[i]:last_upright[i]=info['upright']
  pos=env.gd.geom_xpos.numpy()[:,env.soles];rot=env.gd.geom_xmat.numpy()[:,env.soles].reshape(n,2,3,3)
  z=pos[:,leg,2,None]+rot[:,leg,2,:]@verts[leg].T;low=z.min(axis=1)
  support=np.array([x['upright']>np.cos(np.deg2rad(10)) and x['sampled_sole_peak_normal_n'][1-leg]>.5 for x in infos])&alive
  best=np.maximum(best,np.where(support,low,0));counts+=support&(low>.001)
 out={'seed':5123,'worlds':n,'leg':leg,'frames':frames,'supported_peak_clearance_m':best.tolist(),'supported_frames_over_1mm':counts.tolist(),
      'worlds_with_supported_clearance':int(np.sum(counts>0)),'guard_episodes':int(guards.sum()),'fall_episodes':int(falls.sum()),
      'worlds_with_guard':int(np.count_nonzero(guards)),'guard_count_per_world':guards.tolist(),'final_upright':last_upright.tolist(),
      'worlds_with_clearance_no_guard_and_final_upright':int(np.sum((counts>0)&alive&(last_upright>np.cos(np.deg2rad(10))))),
      'limitations':'One open-loop action sequence over 16 small initial tilts; 20 ms mesh sampling, fixed assumed mechanics. No sustained gait, policy learning or hardware validation.'}
 output=Path(sys.argv[2]) if len(sys.argv)>2 else p/'candidate_robustness_audit.json'
 output.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
