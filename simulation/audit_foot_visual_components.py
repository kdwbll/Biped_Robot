"""Inventory disconnected foot visual mesh components in the current keyframe."""
from pathlib import Path
import json
import mujoco
import numpy as np
import trimesh
p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_smooth_hip_trial.xml'))
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
rows=[]
for leg in (0,1):
 gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'L{leg}_visual6')
 if gid<0:raise RuntimeError('foot geom missing')
 mid=m.geom_dataid[gid];va=m.mesh_vertadr[mid];fa=m.mesh_faceadr[mid]
 mesh=trimesh.Trimesh(vertices=m.mesh_vert[va:va+m.mesh_vertnum[mid]],faces=m.mesh_face[fa:fa+m.mesh_facenum[mid]],process=True)
 rotation=d.geom_xmat[gid].reshape(3,3);parts=[]
 for part in sorted(mesh.split(only_watertight=False),key=lambda x:len(x.faces),reverse=True):
  if len(part.faces)<100:continue
  world=part.vertices@rotation.T+d.geom_xpos[gid]
  parts.append({'face_count':len(part.faces),'local_extent_mm':(part.extents*1000).tolist(),'world_min_mm':(world.min(axis=0)*1000).tolist(),'world_max_mm':(world.max(axis=0)*1000).tolist()})
 rows.append({'geom':f'L{leg}_visual6','components_at_least_100_triangles':parts})
result={'model':'diagnostic_smooth_hip_trial.xml','ground_world_z_mm':0,'feet':rows,'limitations':'Connectivity analysis of pre-existing STL, not a SolidWorks comparison. Components can contain multiple connected solids or separate visible details. Identification of the deleted board requires the user or CAD.'}
(p/'foot_visual_components_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
for r in rows:
 print(r['geom'],[(x['face_count'],np.round(x['world_min_mm'][2],2),np.round(x['world_max_mm'][2],2)) for x in r['components_at_least_100_triangles']])
