import math,json,numpy as np,mujoco
from pathlib import Path
from larger_action_env import LargerActionBipedEnv
rows=[]
for variant in ('neutral','sinusoid'):
 for seed in (7001,7002,7003,7004,7005):
  e=LargerActionBipedEnv(episode_seconds=4);o,_=e.reset(seed=seed);x0=e.data.xipos[e.body,0];peak=0.;clear=np.zeros(2)
  ids=[mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_GEOM,f'L{x}_visual6') for x in (0,1)]
  for i in range(200):
   t=i*.02;u=.6*math.sin(2*math.pi*t);v=.6*math.sin(2*math.pi*t+math.pi/2);action=np.zeros(5) if variant=='neutral' else np.array([u,v,-u,-v,0.]);o,r,term,trunc,info=e.step(action);peak=max(peak,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
   for k,g in enumerate(ids):
    mi=e.model.geom_dataid[g];adr=e.model.mesh_vertadr[mi];n=e.model.mesh_vertnum[mi];world=e.model.mesh_vert[adr:adr+n]@e.data.geom_xmat[g].reshape(3,3).T+e.data.geom_xpos[g];clear[k]=max(clear[k],float(world[:,2].min()))
   if term or trunc:break
  row={'variant':variant,'seed':seed,'distance_m':float(e.data.xipos[e.body,0]-x0),'max_tilt_deg':peak,'switches':info.get('support_switches',0),'max_clearance_m':clear.tolist(),'fell':bool(term),'seconds':(i+1)*.02};rows.append(row);print(row,flush=True)
Path(r'C:\Practice_project\biped_sim\sinusoid_holdout_eval.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
