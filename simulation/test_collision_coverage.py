import mujoco,numpy as np
m=mujoco.MjModel.from_xml_path(r'C:\Practice_project\biped_sim\biped_filtered_collision_experimental.xml');d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);print('initial contacts',d.ncon,'explicit collision pairs',m.npair,'equalities',m.neq)
from collections import Counter
hits=Counter(); penetration=Counter()
for i in range(4000):
 mujoco.mj_step(m,d)
 for c in d.contact[:d.ncon]:
  a=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or '';b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or ''
  k='ground' if 'ground' in (a,b) else ('shaft' if a.startswith('visual_pin_') or b.startswith('visual_pin_') else ('upper' if a.startswith('upper_') or b.startswith('upper_') else 'other')); hits[k]+=1;penetration[k]=max(penetration[k],-c.dist*1000)
print('contact frames by type',dict(hits),'max penetration mm',dict(penetration),'finite',np.isfinite(d.qpos).all())

