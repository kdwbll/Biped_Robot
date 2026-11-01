"""Measure high-resolution chassis proxies against refined tower proxies at rest."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
tree=ET.parse(p/'diagnostic_post_shaft_fix_gear_phase_25.xml');root=tree.getroot()
parent=next(b for b in root.iter('body') if b.find("geom[@name='dc_chassis_5_0']") is not None)
assets=root.find('asset');names=[]
for path in sorted((p/'collision_highres'/'chassis5').glob('*.stl')):
 name='highres_'+path.stem;names.append(name)
 ET.SubElement(assets,'mesh',name=name,file='collision_highres/chassis5/'+path.name,scale='.001 .001 .001')
 ET.SubElement(parent,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0')
if not names:raise RuntimeError('Run refine_collision_highres.py chassis5 first')
for leg in (0,1):
 files=sorted((p/'collision_highres'/f'L{leg}').glob('*.stl'))
 body=next(b for b in root.iter('body') if b.get('name')==f'L{leg}_part0')
 for file in files:
  name='highres_'+file.stem
  ET.SubElement(assets,'mesh',name=name,file=f'collision_highres/L{leg}/'+file.name,scale='.001 .001 .001')
  ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0')
root.set('model','diagnostic_highres_gap_only')
path=p/'diagnostic_highres_gap_only.xml';tree.write(path,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
results=[]
for leg in (0,1):
 towers=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
 tower_prefix=f'highres_L{leg}_' if (p/'collision_highres'/f'L{leg}'/'summary.json').exists() else f'refined_L{leg}_tower_'
 towers=[n for n in towers if n and n.startswith(tower_prefix)]
 minimum=1.;worst=None;overlap=0
 for a in names:
  ai=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,a)
  for b in towers:
   bi=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,b)
   distance=float(mujoco.mj_geomDistance(m,d,ai,bi,1.,None))
   if distance<minimum:minimum=distance;worst=[a,b]
   if distance<0:overlap+=1
 results.append({'leg':leg,'tower_proxy_prefix':tower_prefix,'tower_pieces':len(towers),'minimum_signed_distance_mm':minimum*1000,'worst_pair':worst,'overlapping_proxy_pairs':overlap})
out={'model':path.name,'highres_chassis_piece_count':len(names),'results':results,
     'limitations':'Static initial pose only; signed distances are convex proxy approximations, not CAD solid distances.'}
(p/'highres_proxy_gap_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2),flush=True)
