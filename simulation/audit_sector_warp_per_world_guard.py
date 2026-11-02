"""Validate per-world guard termination and resets in Warp VecEnv, no training."""
from pathlib import Path
import json,sys,numpy as np
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import warp as wp
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

rows=[]
env=SectorWarpStandingVecEnv(nworld=4,episode_seconds=1.,on_guard='terminate')
try:
 env.reset();q=env.gd.qpos.numpy();q[2,env.hips[0]]+=np.deg2rad(4.6);wp.copy(env.gd.qpos,wp.array(q,dtype=wp.float32,device='cuda:0'))
 obs,reward,done,info=env.step(np.zeros((4,5),dtype=np.float32));times=env.gd.time.numpy()
 if done.tolist()!=[False,False,True,False] or info[2]['guard_flags']!=6 or reward[2]>=0 or 'terminal_observation' not in info[2] or abs(float(times[2]))>1e-7 or not np.all(times[[0,1,3]]>.019):raise RuntimeError(('hip guard/reset failed',done.tolist(),info,times.tolist()))
 rows.append({'test':'injected_hip','done':done.tolist(),'guard_flags':[x['guard_flags'] for x in info],'world_times_s':times.tolist()})
 v=env.gd.qvel.numpy();v[3,0]=np.nan;wp.copy(env.gd.qvel,wp.array(v,dtype=wp.float32,device='cuda:0'))
 obs,reward,done,info=env.step(np.zeros((4,5),dtype=np.float32));times=env.gd.time.numpy()
 if done.tolist()!=[False,False,False,True] or not (info[3]['guard_flags']&1) or not np.isfinite(obs).all() or not np.isfinite(info[3]['terminal_observation']).all() or abs(float(times[3]))>1e-7:raise RuntimeError(('nonfinite/reset failed',done.tolist(),info,times.tolist()))
 rows.append({'test':'injected_nonfinite','done':done.tolist(),'guard_flags':[x['guard_flags'] for x in info],'world_times_s':times.tolist()})
finally:env.close()
env=SectorWarpStandingVecEnv(nworld=4,episode_seconds=1.,contact_limit_n=.0001,on_guard='terminate')
try:
 env.reset();obs,reward,done,info=env.step(np.zeros((4,5),dtype=np.float32));times=env.gd.time.numpy();guard=np.array([x['guard_flags'] for x in info])
 if not np.any((guard&8)>0) or not np.all(done[(guard&8)>0]) or not np.isfinite(obs).all():raise RuntimeError(('live contact guard failed',done.tolist(),guard.tolist()))
 rows.append({'test':'live_tiny_contact_threshold','done':done.tolist(),'guard_flags':guard.tolist(),'world_times_s':times.tolist()})
finally:env.close()
out={'results':rows,'limitations':'Three short tests with injected states or intentionally tiny contact threshold; termination freezes/resets selected worlds within a control frame, while overflow/time-reset still aborts the batch. Sampled force can miss spikes, no training or hardware.'}
(p/'sector_warp_per_world_guard_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
