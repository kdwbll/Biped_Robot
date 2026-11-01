import mujoco,numpy as np
m=mujoco.MjModel.from_xml_path(r'C:\Practice_project\biped_sim\biped_chassis_collision_candidate.xml');d=mujoco.MjData(m);ground=0;selfp=0;links=0;knees=set()
for i in range(4000):
 mujoco.mj_step(m,d)
 for c in d.contact[:d.ncon]:
  a=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or '';b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or ''
  if 'ground' in (a,b):ground=max(ground,-c.dist)
  if (a.startswith(('col_','knee_','hull_')) and b.startswith(('col_','knee_','hull_'))):
   selfp=max(selfp,-c.dist)
   if a.startswith('knee_') or b.startswith('knee_'):knees.add(tuple(sorted((a,b))))
 for k in range(d.nefc):
  if d.efc_type[k]==mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[k]<16:links=max(links,abs(d.efc_pos[k]))
print('seconds',d.time,'finite',np.isfinite(d.qpos).all(),'ground mm',ground*1000,'self mm',selfp*1000,'link mm',links*1000,'knee contacts',len(knees))

