"""Verify compiled MuJoCo pinion pose matches intended CAD phase rotation."""
from pathlib import Path
import json
import sys
import math
import mujoco
import numpy as np
from scipy.spatial import cKDTree
import trimesh

p=Path(__file__).parent;phase25='--25-tooth' in sys.argv
m=mujoco.MjModel.from_xml_path(str(p/('diagnostic_post_shaft_fix_gear_phase_25.xml' if phase25 else 'diagnostic_post_shaft_fix_gear_phase.xml')))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'phase_pinion_visual')
b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')
mesh_id=m.geom_dataid[g];verts=m.mesh_vert[m.mesh_vertadr[mesh_id]:m.mesh_vertadr[mesh_id]+m.mesh_vertnum[mesh_id]]
world=verts@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
actual=(world-d.xpos[b])@d.xmat[b].reshape(3,3)
source=trimesh.load_mesh(p/'gear_phase_visuals'/'pinion_local.stl',process=True).vertices*.001
q=d.qpos[m.jnt_qposadr[j]];c=math.cos(q);s=math.sin(q);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
center=np.array([-.0266628795,-.006474586,.26632]);expected=source@R.T+center
da=cKDTree(expected).query(actual)[0];db=cKDTree(actual).query(expected)[0]
err=float(max(max(da),max(db)))
if err>1e-5:raise RuntimeError(f'Visual gear pose differs by {err} m')
out={'max_nearest_vertex_error_m':err,'shaft_angle_deg':math.degrees(q),'compiled_mesh_vertices':len(actual),
     'source_mesh_vertices':len(expected),'note':'Checks visual pose only, not gear contact physics.'}
(p/('gear_phase_visual_25_audit.json' if phase25 else 'gear_phase_visual_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
print('Pinion visual position matches phase-adjusted shaft; max vertex difference mm',err*1000)
