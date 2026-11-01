"""Try local refined collision proxies; retain excluded regions as exclusions.

Diagnostic only. Mapping convex pieces by overlap is conservative, and no
physical-collision coverage is claimed from this model.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import scipy.spatial
import trimesh
import mujoco

p=Path(__file__).parent
contacts=json.loads((p/'standing_contact_forces_audit.json').read_text(encoding='utf8'))
suspects={tuple(sorted(r['geoms'])) for r in contacts if 'ground' not in r['geoms'] and r['normal_force_n']>100.}
def pair_set(file):
 return {tuple(sorted((x.get('geom1'),x.get('geom2')))) for x in ET.parse(p/file).getroot().find('contact').findall('pair')}
excluded=pair_set('biped_decomposed_chassis_trial.xml')-pair_set('biped_filtered_collision_experimental.xml')
tree=ET.parse(p/'biped_standing_gear_candidate.xml');root=tree.getroot();assets=root.find('asset');pairs=root.find('contact')
old_by_leg={leg:sorted((p/'tower_collision_pieces').glob(f'L{leg}_tower_*.stl')) for leg in (0,1)}
new_by_leg={leg:sorted((p/'tower_collision_refined').glob(f'L{leg}_tower_*.stl')) for leg in (0,1)}
def inside(vertices,mesh):
 hull=scipy.spatial.ConvexHull(mesh.vertices)
 return bool(np.any(np.all(vertices@hull.equations[:,:3].T+hull.equations[:,3]<=1e-6,axis=1)))
mapping={}
for leg in (0,1):
 old={f'tower_{x.stem}':trimesh.load_mesh(x,process=True) for x in old_by_leg[leg]}
 for f in new_by_leg[leg]:
  new=trimesh.load_mesh(f,process=True)
  body=root.find(f".//body[@name='L{leg}_part0']")
  name='refined_'+f.stem
  ET.SubElement(assets,'mesh',name=name,file='tower_collision_refined/'+f.name,scale='.001 .001 .001')
  ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0')
  overlap=[]
  for old_name,mesh in old.items():
   if np.any(new.bounds[1]<mesh.bounds[0]) or np.any(mesh.bounds[1]<new.bounds[0]):continue
   cm=trimesh.collision.CollisionManager();cm.add_object('old',mesh)
   if inside(new.vertices,mesh) or inside(mesh.vertices,new) or cm.in_collision_single(new):overlap.append(old_name)
  mapping[name]=overlap
added=set();removed=[];skipped=0
for pair in list(pairs.findall('pair')):
 key=tuple(sorted((pair.get('geom1'),pair.get('geom2'))))
 if key not in suspects:continue
 pairs.remove(pair);removed.append(key)
 chassis=next(x for x in key if x.startswith('dc_chassis_'))
 tower=next(x for x in key if x.startswith('tower_'))
 for name,old_towers in mapping.items():
  if tower not in old_towers:continue
  if any(tuple(sorted((chassis,x))) in excluded for x in old_towers):skipped+=1;continue
  newkey=tuple(sorted((chassis,name)))
  if newkey in added:continue
  attrs=dict(pair.attrib);attrs.pop('name',None);attrs['geom1']=chassis;attrs['geom2']=name
  ET.SubElement(pairs,'pair',attrs);added.add(newkey)
if set(removed)!=suspects:raise RuntimeError('Suspect pair missing')
root.set('model','diagnostic_refined_tower');ET.indent(root)
out=p/'diagnostic_refined_tower.xml';tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
result={'removed':len(removed),'added':len(added),'skipped_due_to_preexisting_exclusions':skipped,
        'new_pairs':m.npair,'initial_contacts':d.ncon,'mapping':mapping}
(p/'refined_tower_build_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('removed',len(removed),'refined added',len(added),'skipped (preexisting exclusions)',skipped,
      'total pairs',m.npair,'initial contacts',d.ncon)
