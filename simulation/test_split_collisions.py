import numpy as np,mujoco
m=mujoco.MjModel.from_xml_path(r'C:\Practice_project\biped_sim\self_collision_split_candidate.xml');d=mujoco.MjData(m);maxfloor=0;maxself=0;seen=set()
for i in range(4000):
 mujoco.mj_step(m,d)
 for c in d.contact[:d.ncon]:
  a=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or '';b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or ''
  if 'ground' in (a,b):maxfloor=max(maxfloor,-c.dist)
  elif a.startswith('col_') and b.startswith('col_'):maxself=max(maxself,-c.dist);seen.add(tuple(sorted((a,b))))
 if not np.isfinite(d.qpos).all():break
print('time',d.time,'max ground penetration mm',maxfloor*1000,'max self penetration mm',maxself*1000,'different self pairs',len(seen),'finite',np.isfinite(d.qpos).all())
