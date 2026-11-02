"""Compare neutral pauses between alternating simulated lift sequences."""
from pathlib import Path
import json
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent
h1=np.asarray(json.loads((p/'训练记录/lift_search_20261102_a/best.json').read_text(encoding='utf8'))['action_history'])
h0=np.asarray(json.loads((p/'训练记录/opposite_lift_20261103_a/best.json').read_text(encoding='utf8'))['action_history'])
delays=np.repeat([0,25,50,100],4);n=len(delays);env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=14.,on_guard='terminate',reset_tilt_deg=1.,pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 env.reset_rng=[np.random.default_rng(14001+i%4) for i in range(n)];env.reset()
 verts=[]
 for geom in env.soles:
  mesh=int(env.m.geom_dataid[geom]);start=int(env.m.mesh_vertadr[mesh]);count=int(env.m.mesh_vertnum[mesh]);verts.append(env.m.mesh_vert[start:start+count].copy())
 counts=np.zeros((n,4),int);peaks=np.zeros((n,4));guards=np.zeros(n,int);falls=np.zeros(n,int);alive=np.ones(n,bool);finished=np.zeros(n,bool);last_upright=np.ones(n);start_xy=env.gd.qpos.numpy()[:,:2].copy();last_xy=start_xy.copy()
 duration=360+3*delays
 for step in range(int(max(duration))):
  action=np.zeros((n,5));phases=np.full(n,-1,int)
  for i,delay in enumerate(delays):
   if step>=duration[i]:finished[i]=True;continue
   phase=step//(90+delay);local=step%(90+delay)
   if phase<4 and local<90:phases[i]=phase;action[i]=[h0,h1][phase%2][local//30]
  _,_,done,infos=env.step(action)
  for i,info in enumerate(infos):
   if not alive[i] or finished[i]:continue
   if done[i] and info['guard_flags']:guards[i]+=1;alive[i]=False
   elif done[i] and not info['TimeLimit.truncated']:falls[i]+=1;alive[i]=False
   if alive[i]:last_upright[i]=info['upright']
  q=env.gd.qpos.numpy();track=alive&~finished;last_xy[track]=q[track,:2]
  pos=env.gd.geom_xpos.numpy()[:,env.soles];rot=env.gd.geom_xmat.numpy()[:,env.soles].reshape(n,2,3,3)
  for leg in (0,1):
   low=(pos[:,leg,2,None]+rot[:,leg,2,:]@verts[leg].T).min(axis=1)
   supported=np.array([x['upright']>np.cos(np.deg2rad(10)) and x['sampled_sole_peak_normal_n'][1-leg]>.5 for x in infos])&alive&~finished&(phases>=0)&(phases%2==leg)
   for i in np.flatnonzero(supported):
    phase=phases[i];peaks[i,phase]=max(peaks[i,phase],low[i]);counts[i,phase]+=low[i]>.001
 safe=alive&(last_upright>np.cos(np.deg2rad(10)))
 out={'delays_frames':delays.tolist(),'worlds_per_delay':4,'supported_peak_clearance_m':peaks.tolist(),'supported_frames_over_1mm':counts.tolist(),'guard_count_per_world':guards.tolist(),'fall_count_per_world':falls.tolist(),'final_upright':last_upright.tolist(),'body_xy_displacement_m':(last_xy-start_xy).tolist(),'all_four_phases_safe_by_delay':[int(np.sum(safe[sl]&np.all(counts[sl]>0,axis=1))) for sl in (slice(0,4),slice(4,8),slice(8,12),slice(12,16))], 'limitations':'Four small randomized poses shared across each pause length. Neutral action is a target offset, not motor-off. Four short lift phases; only 20 ms samples. Assumed dynamics, guards, and no hardware validation.'}
 outpath=p/'训练记录/opposite_lift_recovery_20261103_a.json';outpath.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
