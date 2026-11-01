import itertools,numpy as np,mujoco
from larger_action_env import LargerActionBipedEnv
for leg in (0,1):
 for a,b in itertools.product((-1.,0.,1.),repeat=2):
  if a==b==0:continue
  e=LargerActionBipedEnv(episode_seconds=2,disturbance=0);o,_=e.reset(seed=0);act=np.zeros(5);act[2*leg:2*leg+2]=[a,b];g=mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_GEOM,f'L{leg}_visual6');mi=e.model.geom_dataid[g];k=e.model.mesh_vertadr[mi];n=e.model.mesh_vertnum[mi];peak=-1
  for i in range(100):
   o,r,term,trunc,info=e.step(act);v=e.model.mesh_vert[k:k+n]@e.data.geom_xmat[g].reshape(3,3).T+e.data.geom_xpos[g];peak=max(peak,float(v[:,2].min()))
   if term or trunc:break
  print('leg',leg,'action',a,b,'max lift mm',round(peak*1000,2),'tilt',round(float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))),1),'fell',term,flush=True)
