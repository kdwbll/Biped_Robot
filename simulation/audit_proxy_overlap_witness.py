"""Locate overlap witness points relative to the original source surfaces."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import trimesh

p=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_nomerge_full_gap_only.xml'));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
src=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL')),process=True)
parts=[x for x in src.split(only_watertight=False,repair=False) if x.is_watertight and len(x.faces)>=20 and x.convex_hull.volume>=.0001]
chassis=parts[5];chassis.apply_scale(.001)
towers=[]
for filename in ('single_leg_mesh_experiment.xml','second_leg_mesh_experiment.xml'):
 root=ET.parse(p/filename).getroot()
 pieces=trimesh.load_mesh(p/root.find("./asset/mesh[@name='mesh0']").get('file'),process=True).split(only_watertight=False,repair=False)
 tower=max((x for x in pieces if x.is_watertight and x.extents[2]>100),key=lambda x:len(x.faces));tower.apply_scale(.001);towers.append(tower)
gap=json.loads((p/'nomerge_full_proxy_gap_audit.json').read_text(encoding='utf8'));rows=[]
for entry in gap['results']:
 a,b=entry['worst_pair'];ai=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,a);bi=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,b)
 witness=np.zeros(6);distance=float(mujoco.mj_geomDistance(m,d,ai,bi,1.,witness))
 distances=[]
 for point,geom,source in ((witness[:3],ai,chassis),(witness[3:],bi,towers[entry['leg']])):
  body=m.geom_bodyid[geom];local=(point-d.xpos[body])@d.xmat[body].reshape(3,3)
  _,dist,_=trimesh.proximity.closest_point_naive(source,local[None,:]);distances.append(float(dist[0])*1000)
 rows.append({'leg':entry['leg'],'proxy_pair':[a,b],'proxy_signed_distance_mm':distance*1000,
  'witness_world_m':witness.tolist(),'witness_to_source_surface_mm':distances})
out={'results':rows,'limitations':'Closest surface distances do not determine inside/outside or CAD clearance; witness positions are collision-algorithm outputs for overlapping convex approximations.'}
(p/'proxy_overlap_witness_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
