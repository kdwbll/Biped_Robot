import numpy as np,mujoco
m=mujoco.MjModel.from_xml_path(r'C:\Practice_project\biped_sim\biped_estimated_mass_collision.xml');d=mujoco.MjData(m);floor=0;selfp=0;eq=0
for i in range(4000):
 mujoco.mj_step(m,d)
 for c in d.contact[:d.ncon]:
  a=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or '';b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or ''
  if 'ground' in (a,b):floor=max(floor,-c.dist)
  if a.startswith('col_') and b.startswith('col_'):selfp=max(selfp,-c.dist)
 for k in range(d.nefc):
  if d.efc_type[k]==mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[k]<16:eq=max(eq,abs(d.efc_pos[k]))
print('2s finite',np.isfinite(d.qpos).all(),'ground mm',floor*1000,'enabled self mm',selfp*1000,'joint error mm',eq*1000)
