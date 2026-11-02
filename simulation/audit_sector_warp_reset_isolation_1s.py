"""Compare one-second unaffected GPU worlds with and without another world reset."""
from pathlib import Path
import gc,json,sys,numpy as np
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import warp as wp
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv
rng=np.random.default_rng(46);actions=np.repeat(rng.uniform(-.1,.1,(5,4,5)),10,axis=0).astype(np.float32);keep=[0,1,3];sample_steps=(9,10,24,49)
def run(inject):
 e=SectorWarpStandingVecEnv(nworld=4,episode_seconds=2.,on_guard='terminate');states=[];reset_seen=False
 try:
  e.reset()
  for step,act in enumerate(actions):
   if inject and step==10:
    q=e.gd.qpos.numpy();q[2,e.hips[0]]+=np.deg2rad(4.6);wp.copy(e.gd.qpos,wp.array(q,dtype=wp.float32,device='cuda:0'))
   _,_,done,info=e.step(act)
   if inject and step==10:
    if done.tolist()!=[False,False,True,False] or not (info[2]['guard_flags']&2):raise RuntimeError('Expected reset did not occur')
    reset_seen=True
   elif done.any():raise RuntimeError(('Unexpected termination',step,done.tolist()))
   if step in sample_steps:states.append({'step':step,'q':e.gd.qpos.numpy()[keep].copy(),'v':e.gd.qvel.numpy()[keep].copy()})
  return states,reset_seen
 finally:e.close();del e;gc.collect()
a,_=run(False);b,_=run(False);c,seen=run(True);rows=[]
for x,y,z in zip(a,b,c):
 rows.append({'step':x['step']+1,'repeat_qpos_max_difference':float(np.max(np.abs(x['q']-y['q']))),'repeat_qvel_max_difference':float(np.max(np.abs(x['v']-y['v']))),'reset_qpos_max_difference':float(np.max(np.abs(x['q']-z['q']))),'reset_qvel_max_difference':float(np.max(np.abs(x['v']-z['v'])))})
out={'reset_seen':seen,'results':rows,'limitations':'One fixed low-amplitude action sequence, four GPU worlds for one second. Repeat differences include numerical nondeterminism; this does not prove independence over all states or multi-episode training.'}
(p/'sector_warp_reset_isolation_1s_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
