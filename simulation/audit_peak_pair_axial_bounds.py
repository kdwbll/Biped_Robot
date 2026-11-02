"""Check axial intervals and rigid ancestry for a suspected redundant contact."""
from pathlib import Path
import json
import mujoco
import numpy as np
p=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_collision_trial.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');hip=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'L0_hip_candidate');axis=d.xaxis[hip].copy();origin=d.xanchor[hip].copy();mount=m.jnt_bodyid[hip];names=['dc_chassis_1_1','tower_L0_tower_8'];rows=[]
for n in names:
 g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n);body=m.geom_bodyid[g];target=root if n==names[0] else mount
 while body!=target:
  if body<=0 or m.body_jntnum[body]:raise RuntimeError('Proxy ancestry has additional degrees of freedom')
  body=m.body_parentid[body]
 mid=m.geom_dataid[g];addr=m.mesh_vertadr[mid];v=m.mesh_vert[addr:addr+m.mesh_vertnum[mid]]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
 axial=(v-origin)@axis;rows.append({'geom':n,'axial_min_mm':float(axial.min()*1000),'axial_max_mm':float(axial.max()*1000)})
gap=rows[0]['axial_min_mm']-rows[1]['axial_max_mm']
if abs(gap)>.001:raise RuntimeError(('Expected near-coplanar axial bounds',gap))
out={'model':'diagnostic_sector_collision_trial.xml','results':rows,'axial_gap_mm':gap,'limitations':'Bounds and ancestry reflect the ideal model, not CAD manufacturing or real contact. Axial hinge prohibits relative separation as well as penetration; tangential friction/load not represented by removing this contact.'}
(p/'peak_pair_axial_bounds_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
