"""Check two repeated cycles of open-loop lift schedules without resets."""
from pathlib import Path
import json
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent
h1=np.asarray(json.loads((p/'训练记录/lift_search_20261102_a/best.json').read_text(encoding='utf8'))['action_history'])
h0=np.asarray(json.loads((p/'训练记录/opposite_lift_20261103_a/best.json').read_text(encoding='utf8'))['action_history'])
n=16; env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=8.,on_guard='terminate',reset_tilt_deg=1.,pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
 env.reset_rng=[np.random.default_rng(12201+i%8) for i in range(n)];env.reset()
 verts=[]
 for geom in env.soles:
  mesh=int(env.m.geom_dataid[geom]);a=int(env.m.mesh_vertadr[mesh]);count=int(env.m.mesh_vertnum[mesh]);verts.append(env.m.mesh_vert[a:a+count].copy())
 start_xy=env.gd.qpos.numpy()[:,:2].copy();last_xy=start_xy.copy();peaks=np.zeros((n,4));frames=np.zeros((n,4),int);guards=np.zeros(n,int);falls=np.zeros(n,int);alive=np.ones(n,bool);upright=np.ones(n)
 for step in range(360):
  phase=step//90; order_phase=phase%2; j=(step%90)//30; action=np.empty((n,5));action[:8]=[h1,h0][order_phase][j];action[8:]=[h0,h1][order_phase][j]
  _,_,done,infos=env.step(action)
  for i,info in enumerate(infos):
   if done[i] and info['guard_flags'] and alive[i]:guards[i]+=1;alive[i]=False
   elif done[i] and not info['TimeLimit.truncated'] and alive[i]:falls[i]+=1;alive[i]=False
   if alive[i]:upright[i]=info['upright']
  qpos=env.gd.qpos.numpy();last_xy[alive]=qpos[alive,:2]
  pos=env.gd.geom_xpos.numpy()[:,env.soles];rot=env.gd.geom_xmat.numpy()[:,env.soles].reshape(n,2,3,3)
  for leg in (0,1):
   target=np.array([([1,0] if i<8 else [0,1])[order_phase]==leg for i in range(n)])
   low=(pos[:,leg,2,None]+rot[:,leg,2,:]@verts[leg].T).min(axis=1)
   support=np.array([x['upright']>np.cos(np.deg2rad(10)) and x['sampled_sole_peak_normal_n'][1-leg]>.5 for x in infos])&alive&target
   peaks[:,phase]=np.maximum(peaks[:,phase],np.where(support,low,0));frames[:,phase]+=support&(low>.001)
 safe=alive&(upright>np.cos(np.deg2rad(10)))
 out={'seed_family':'12201+i%8','worlds_per_order':8,'frames':360,'orders':['leg1_then_leg0','leg0_then_leg1'],'supported_peak_clearance_m':peaks.tolist(),'supported_frames_over_1mm':frames.tolist(),'guard_count_per_world':guards.tolist(),'fall_count_per_world':falls.tolist(),'final_upright':upright.tolist(),'all_four_phases_safe_by_order':[int(np.sum(safe[sl] & np.all(frames[sl]>0,axis=1))) for sl in (slice(0,8),slice(8,16))], 'body_xy_displacement_m':(last_xy-start_xy).tolist(),'limitations':'Repeated open-loop schedules for 7.2 s in 16 small initial tilts; 20 ms mesh samples, assumed mechanics and contact guard. No proof of forward steps, sustained walking, or hardware safety.'}
 outpath=p/'训练记录/opposite_lift_20261103_a/alternating_two_cycles_probe.json';outpath.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
