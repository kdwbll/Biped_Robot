"""Create a foot geometry/mass trial after read-only confirmation in SolidWorks."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import trimesh
p=Path(__file__).parent
src=p/'meshes_second_leg_local/part6.stl'
mesh=trimesh.load_mesh(src,process=True)
parts=sorted(mesh.split(only_watertight=False),key=lambda a:len(a.faces),reverse=True)
if len(parts)!=3 or [len(x.faces) for x in parts]!=[2222,1394,628]:raise RuntimeError('Unexpected source mesh; do not remove an unidentified component')
removed=parts[-1]
if not removed.is_watertight or not np.allclose(sorted(removed.extents),[2,10,125],atol=.1):raise RuntimeError('Unrecognized plate')
new_mesh=p/'meshes_second_leg_local/part6_suppressed_plate_trial.stl'
trimesh.util.concatenate(parts[:2]).export(new_mesh)
inv=json.loads((p/'cad_mass_inventory.json').read_text(encoding='utf8'))
plate=[a for a in inv if a['body']=='L1_part6' and np.allclose(sorted(a['box_size_m']),[.002,.010,.125],atol=.0001)]
if len(plate)!=1:raise RuntimeError('Plate mass entry not unique')
plate=plate[0]
tree=ET.parse(p/'diagnostic_smooth_hip_trial.xml');root=tree.getroot()
root.find(".//mesh[@name='L1_mesh6']").set('file','meshes_second_leg_local/'+new_mesh.name)
it=root.find(".//body[@name='L1_part6']/inertial")
oldmass=float(it.get('mass'));oldc=np.fromstring(it.get('pos'),sep=' ');v=np.fromstring(it.get('fullinertia'),sep=' ')
I=np.array([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]]);pm=plate['mass_kg'];pc=np.array(plate['center_m']);PI=np.array(plate['approx_inertia_at_center'])
newmass=oldmass-pm;newc=(oldmass*oldc-pm*pc)/newmass
def shift(m,r):return m*((r@r)*np.eye(3)-np.outer(r,r))
newI=I-shift(oldmass,oldc-newc)-PI-shift(pm,pc-newc)
if min(np.linalg.eigvalsh(newI))<=0:raise RuntimeError('invalid inertia')
it.set('mass',f'{newmass:.12g}');it.set('pos',' '.join(f'{x:.12g}' for x in newc));it.set('fullinertia',' '.join(f'{x:.12g}' for x in [newI[0,0],newI[1,1],newI[2,2],newI[0,1],newI[0,2],newI[1,2]]))
root.set('model','diagnostic_foot_suppressed_plate_trial');out=p/'diagnostic_foot_suppressed_plate_trial.xml';tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
report={'old_mass_kg':oldmass,'removed_plate_mass_kg':pm,'new_mass_kg':newmass,'removed_triangles':len(removed.faces),'remaining_triangles':sum(len(x.faces) for x in parts[:2]),'contact_pairs':int(m.npair),'limitations':'Two visible CAD bodies matched by inventory, not re-exported CAD tessellation. Mesh-based contact may be convexified. Inertia uses old box estimates. Remaining exclusions and unmeasured actuator parameters persist; diagnostic only.'}
(p/'foot_suppressed_plate_trial_build.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
