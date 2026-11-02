"""Use separate sole and upper foot mesh components for ground contact."""
from pathlib import Path
import json
import copy
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import trimesh
p=Path(__file__).parent;source=p/'diagnostic_foot_suppressed_plate_trial.xml'
tree=ET.parse(source);root=tree.getroot();assets=root.find('asset');changes=[]
for leg in (0,1):
 visual=root.find(f"./asset/mesh[@name='L{leg}_mesh6']");mesh=trimesh.load_mesh(p/visual.get('file'),process=True)
 parts=mesh.split(only_watertight=False)
 candidates=[q for q in parts if q.is_watertight and len(q.faces)>1000 and np.allclose(sorted(q.extents),sorted([125,49 if leg else 50,25]),atol=2)]
 if len(candidates)!=1:raise RuntimeError(f'Expected unique sole component leg {leg}, got {len(candidates)}')
 sole=candidates[0];rel=f'meshes_foot_soles/leg{leg}_sole_trial.stl';path=p/rel;path.parent.mkdir(exist_ok=True);sole.export(path)
 assetname=f'L{leg}_sole_trial_mesh';geomname=f'L{leg}_sole_trial_geom'
 ET.SubElement(assets,'mesh',name=assetname,file=rel,scale='.001 .001 .001')
 body=root.find(f".//body[@name='L{leg}_part6']")
 ET.SubElement(body,'geom',name=geomname,type='mesh',mesh=assetname,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0',group='3')
 pair=root.find(f"./contact/pair[@name='floor_L{leg}_6']")
 if pair is None or pair.get('geom2')!=f'L{leg}_visual6':raise RuntimeError('Unexpected ground contact')
 pair.set('geom2',geomname)
 upper=[q for q in parts if q.is_watertight and len(q.faces)>1000 and np.allclose(sorted(q.extents),sorted([125,67 if leg else 68,29.75]),atol=2)]
 if len(upper)!=1:raise RuntimeError(f'Expected unique upper foot component leg {leg}')
 upperrel=f'meshes_foot_soles/leg{leg}_upper_trial.stl';upper[0].export(p/upperrel)
 upperasset=f'L{leg}_foot_upper_trial_mesh';uppergeom=f'L{leg}_foot_upper_trial_geom'
 ET.SubElement(assets,'mesh',name=upperasset,file=upperrel,scale='.001 .001 .001')
 ET.SubElement(body,'geom',name=uppergeom,type='mesh',mesh=upperasset,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0',group='3')
 upperpair=copy.deepcopy(pair);upperpair.set('name',f'floor_L{leg}_upper_trial');upperpair.set('geom2',uppergeom);root.find('contact').append(upperpair)
 changes.append({'leg':leg,'mesh':rel,'face_count':len(sole.faces),'extents_mm':sole.extents.tolist(),'pair':pair.get('name'),'upper_mesh':upperrel,'upper_faces':len(upper[0].faces)})
root.set('model','diagnostic_separate_sole_contact_trial');out=p/'diagnostic_separate_sole_contact_trial.xml';tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
report={'source':source.name,'model':out.name,'changes':changes,'compiled_pair_count':int(m.npair),'initial_contacts':int(d.ncon),'limitations':'Sole extracted from existing STL, not a new CAD tessellation. Still convex hull of sole, not full concave surface. Initial geometric samples are not dynamic proof. Other excluded contact pairs and actuator uncertainty remain. Diagnostic only, not training or hardware.'}
(p/'separate_sole_contact_build.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
