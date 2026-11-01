import mujoco,numpy as np
from biped_env import BipedEnv
for f in ['biped_filtered_collision_experimental.xml','biped_mirrored_part_centered_experiment.xml']:
 e=BipedEnv(episode_seconds=4,model_path=f);o,_=e.reset();sumforces=np.zeros(2);samples=0
 for k in range(150):
  o,r,term,trunc,info=e.step(np.zeros(5));
  if k<50:continue
  sums=np.zeros(2)
  for n,c in enumerate(e.data.contact[:e.data.ncon]):
   a=mujoco.mj_id2name(e.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or '';b=mujoco.mj_id2name(e.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or ''
   if 'ground' not in (a,b):continue
   other=b if a=='ground' else a
   for side in (0,1):
    if other.startswith('L'+str(side)+'_'):
     out=np.zeros(6);mujoco.mj_contactForce(e.model,e.data,n,out);sums[side]+=out[0]
  sumforces+=sums;samples+=1
  if term or trunc:break
 print(f,'mean normal foot/leg contact N',np.round(sumforces/max(samples,1),3),'samples',samples)
