"""Add non-colliding sector meshes for pairwise distance diagnostics only."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import mujoco
p=Path(__file__).parent;data=json.loads((p/'sector_clip_prototype_audit.json').read_text(encoding='utf8'))
tree=ET.parse(p/'diagnostic_restored_near_pairs_trial.xml');root=tree.getroot();assets=root.find('asset');added=[]
for row in data['results']:
 old='nomerge_'+row['geom'];parents=[b for b in root.iter('body') if b.find(f"geom[@name='{old}']") is not None]
 if len(parents)!=1:raise RuntimeError((old,len(parents)))
 for filename in row['files']:
  name='sector_'+Path(filename).stem
  ET.SubElement(assets,'mesh',name=name,file='collision_sector_trial/chassis0/'+filename,scale='.001 .001 .001')
  ET.SubElement(parents[0],'geom',name=name,type='mesh',mesh=name,contype='0',conaffinity='0',mass='0',rgba='0 0 0 0',group='3')
  added.append(name)
root.set('model','diagnostic_sector_gap_only');out=p/'diagnostic_sector_gap_only.xml';tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
report={'model':out.name,'added_inactive_geometry_count':len(added),'compiled_contact_pair_count':int(m.npair),'limitations':'Adds non-colliding geometry for distance measurements only. Existing collisions are unchanged. Sector seams and dynamic effects have not been tested; not for training or hardware.'}
(p/'sector_gap_only_build.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
