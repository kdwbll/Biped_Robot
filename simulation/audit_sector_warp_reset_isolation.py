"""Measure impact on other GPU worlds when one guarded world is reset."""
from pathlib import Path
import gc,json,sys,numpy as np
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import warp as wp
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv
act=np.zeros((4,5),dtype=np.float32)
e=SectorWarpStandingVecEnv(nworld=4,episode_seconds=1.,on_guard='terminate')
try:
 e.reset();e.step(act);baseq=e.gd.qpos.numpy().copy();basev=e.gd.qvel.numpy().copy()
finally:e.close();del e;gc.collect()
e=SectorWarpStandingVecEnv(nworld=4,episode_seconds=1.,on_guard='terminate')
try:
 e.reset();e.step(act);repeatq=e.gd.qpos.numpy().copy();repeatv=e.gd.qvel.numpy().copy()
finally:e.close();del e;gc.collect()
e=SectorWarpStandingVecEnv(nworld=4,episode_seconds=1.,on_guard='terminate')
try:
 e.reset();q=e.gd.qpos.numpy();q[2,e.hips[0]]+=np.deg2rad(4.6);wp.copy(e.gd.qpos,wp.array(q,dtype=wp.float32,device='cuda:0'));_,_,done,infos=e.step(act)
 if done.tolist()!=[False,False,True,False]:raise RuntimeError('Unexpected guard termination')
 q=e.gd.qpos.numpy();v=e.gd.qvel.numpy();keep=[0,1,3]
 out={'repeat_qpos_max_difference':float(np.max(np.abs(repeatq[[0,1,3]]-baseq[[0,1,3]]))),'repeat_qvel_max_difference':float(np.max(np.abs(repeatv[[0,1,3]]-basev[[0,1,3]]))),'other_world_qpos_max_difference':float(np.max(np.abs(q[keep]-baseq[keep]))),'other_world_qvel_max_difference':float(np.max(np.abs(v[keep]-basev[keep]))),'guarded_world':2,'step_duration_s':.02,'limitations':'Repeated and injected neutral control steps; float32 nondeterminism and contact ordering can affect differences. Not proof of isolation over long trajectories or training.'}
 (p/'sector_warp_reset_isolation_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:e.close()
