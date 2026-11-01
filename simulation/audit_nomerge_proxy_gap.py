"""Compare unmerged chassis proxies with existing high-resolution leg proxies."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
folder=p/'collision_nomerge'/'chassis5';meta=json.loads((folder/'summary.json').read_text(encoding='utf8'))
files=sorted(folder.glob('*.stl'))
if len(files)!=meta['output_count']:raise RuntimeError('Incomplete CoACD output')
tree=ET.parse(p/'diagnostic_highres_gap_only.xml');root=tree.getroot();assets=root.find('asset')
parent=next(b for b in root.iter('body') if b.find("geom[@name='dc_chassis_5_0']") is not None)
new=[]
for f in files:
 name='nomerge_'+f.stem;new.append(name)
 ET.SubElement(assets,'mesh',name=name,file='collision_nomerge/chassis5/'+f.name,scale='.001 .001 .001')
 ET.SubElement(parent,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0')
root.set('model','diagnostic_nomerge_proxy_gap_only');path=p/'diagnostic_nomerge_proxy_gap_only.xml'
tree.write(path,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
rows=[]
for leg in (0,1):
 towers=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
 towers=[n for n in towers if n and n.startswith(f'highres_L{leg}_')]
 if len(towers)!=96:raise RuntimeError('Expected 96 leg proxies')
 minimum=1.;count=0;worst=None
 for a in new:
  ai=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,a)
  for b in towers:
   bi=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,b)
   gap=float(mujoco.mj_geomDistance(m,d,ai,bi,.001,None))
   if gap<minimum:minimum=gap;worst=[a,b]
   if gap<0:count+=1
 rows.append({'leg':leg,'minimum_signed_distance_mm':minimum*1000,'overlapping_pairs':count,'worst_pair':worst})
out={'chassis_proxy_count':len(new),'tower_proxy_count_per_leg':96,'results':rows,
 'limitations':'Initial pose only, unchanged tower proxies; convex-proxy gaps are not exact CAD clearances; model adds no contact pairs.'}
(p/'nomerge_proxy_gap_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2),flush=True)
