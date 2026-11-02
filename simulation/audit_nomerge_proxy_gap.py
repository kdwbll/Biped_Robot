"""Compare unmerged chassis proxies with existing high-resolution leg proxies."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
finer='--finer' in sys.argv
chassis_folder='collision_finer' if finer else 'collision_nomerge'
folder=p/chassis_folder/'chassis5';meta=json.loads((folder/'summary.json').read_text(encoding='utf8'))
files=sorted(folder.glob('*.stl'))
if len(files)!=meta['output_count']:raise RuntimeError('Incomplete CoACD output')
tree=ET.parse(p/'diagnostic_highres_gap_only.xml');root=tree.getroot();assets=root.find('asset')
legs='--legs' in sys.argv
parent=next(b for b in root.iter('body') if b.find("geom[@name='dc_chassis_5_0']") is not None)
new=[]
for f in files:
 name='nomerge_'+f.stem;new.append(name)
 ET.SubElement(assets,'mesh',name=name,file=chassis_folder+'/chassis5/'+f.name,scale='.001 .001 .001')
 ET.SubElement(parent,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0')
if legs:
 for leg in (0,1):
  key=f'L{leg}';folder=p/'collision_nomerge'/key
  legmeta=json.loads((folder/'summary.json').read_text(encoding='utf8'))
  legfiles=sorted(folder.glob('*.stl'))
  if len(legfiles)!=legmeta['output_count']:raise RuntimeError('Incomplete leg proxy set')
  body=next(x for x in root.iter('body') if x.get('name')==f'{key}_part0')
  for f in legfiles:
   name='nomerge_'+f.stem
   ET.SubElement(assets,'mesh',name=name,file=f'collision_nomerge/{key}/'+f.name,scale='.001 .001 .001')
   ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0')
root.set('model','diagnostic_nomerge_proxy_gap_only');path=p/'diagnostic_nomerge_proxy_gap_only.xml'
if legs:path=p/'diagnostic_nomerge_full_gap_only.xml';root.set('model','diagnostic_nomerge_full_gap_only')
if finer:path=p/'diagnostic_finer_full_gap_only.xml';root.set('model','diagnostic_finer_full_gap_only')
tree.write(path,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
rows=[]
for leg in (0,1):
 towers=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
 prefix=f'nomerge_L{leg}_' if legs else f'highres_L{leg}_'
 towers=[n for n in towers if n and n.startswith(prefix)]
 if not towers:raise RuntimeError('No leg proxies')
 minimum=1.;count=0;worst=None
 for a in new:
  ai=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,a)
  for b in towers:
   bi=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,b)
   gap=float(mujoco.mj_geomDistance(m,d,ai,bi,.001,None))
   if gap<minimum:minimum=gap;worst=[a,b]
   if gap<0:count+=1
 rows.append({'leg':leg,'tower_proxy_count':len(towers),'minimum_signed_distance_mm':minimum*1000,'overlapping_pairs':count,'worst_pair':worst})
out={'chassis_proxy_count':len(new),'results':rows,
 'limitations':'Initial pose only; convex-proxy gaps are not exact CAD clearances; model adds no contact pairs.'}
(p/('finer_full_proxy_gap_audit.json' if finer else 'nomerge_full_proxy_gap_audit.json' if legs else 'nomerge_proxy_gap_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2),flush=True)
