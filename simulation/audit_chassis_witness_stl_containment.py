"""Check where selected proxy witness points lie relative to source STL chassis solid."""
from pathlib import Path
import json
import mujoco
import numpy as np
import trimesh
p=Path(__file__).parent
source=json.loads((p/'remaining_collision_cad_witness_audit.json').read_text(encoding='utf8'))['results']
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_restored_near_pairs_trial.xml'));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');R=d.xmat[root].reshape(3,3);pos=d.xpos[root]
mesh=trimesh.load_mesh(next((p/'meshes_original_coords').glob('*16-40-03-958-1.STL')),process=True)
parts=[c for c in mesh.split(only_watertight=False,repair=False) if c.is_watertight and len(c.faces)>=20 and c.convex_hull.volume>=.0001]
chassis=parts[0]
if not chassis.is_watertight:raise RuntimeError('STL must be watertight for inside test')
points=[]
for x in source:
 a,b=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in x['geoms']];ends=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,a,b,.003,ends)
 if dist<=0:raise RuntimeError('Expected positive proxy separation')
 points.append((ends[:3]-pos)@R*1000)
pts=np.array(points);inside=chassis.contains(pts);signed=trimesh.proximity.signed_distance(chassis,pts)
rows=[]
for item,point,contain,dist in zip(source,pts,inside,signed):
 rows.append({'leg':item['leg'],'geoms':item['geoms'],'cad_nearest_surface_distance_mm':item['chassis_cad_face']['gap_mm'],'point_assembly_mm':point.tolist(),'stl_inside':bool(contain),'stl_signed_distance_mm_positive_inside':float(dist)})
large=[x for x in rows if x['cad_nearest_surface_distance_mm']>.1]
out={'source_stl_faces':len(chassis.faces),'source_stl_watertight':chassis.is_watertight,'total':len(rows),'cad_surface_gap_over_0_1mm':len(large),'large_gap_inside_stl':sum(x['stl_inside'] for x in large),'large_gap_outside_stl':sum(not x['stl_inside'] for x in large),'results':rows,
 'limitations':'STL point containment is approximate, not exact CAD point-in-solid. CAD-to-STL correspondence, sampling, topology, and numerical tolerances remain. Inside is not an error: a convex proxy witness can lie inside the real solid. Does not evaluate dynamic collision or all excluded pairs.'}
(p/'chassis_witness_stl_containment_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print({k:v for k,v in out.items() if k not in ('results','limitations')},flush=True)
