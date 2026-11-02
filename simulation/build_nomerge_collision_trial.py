"""Replace torso-5/tower collision pairs with sampled unmerged proxy pairs."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
tree=ET.parse(p/'diagnostic_nomerge_proxy_gap_only.xml');root=tree.getroot();contact=root.find('contact')
audit=json.loads((p/'nomerge_proxy_range_audit.json').read_text(encoding='utf8'))
excluded={tuple(r['geoms']) for r in audit['near_pairs']}
old_removed=0
for pair in list(contact.findall('pair')):
 a,b=pair.get('geom1'),pair.get('geom2')
 if (a.startswith('dc_chassis_5_') and (b.startswith('tower_') or b.startswith('refined_L'))) or (b.startswith('dc_chassis_5_') and (a.startswith('tower_') or a.startswith('refined_L'))):
  contact.remove(pair);old_removed+=1
names=[g.get('name') for g in root.iter('geom')]
chassis=sorted(n for n in names if n and n.startswith('nomerge_chassis5_'))
towers=sorted(n for n in names if n and (n.startswith('highres_L0_') or n.startswith('highres_L1_')))
if len(chassis)!=387 or len(towers)!=192:raise RuntimeError('Expected complete proxy sets')
added=0
for a in chassis:
 for b in towers:
  if (a,b) in excluded:continue
  ET.SubElement(contact,'pair',geom1=a,geom2=b,solref='.001 1',solimp='.99 .999 .001');added+=1
root.set('model','diagnostic_nomerge_collision_trial')
path=p/'diagnostic_nomerge_collision_trial.xml';tree.write(path,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
out={'old_chassis5_tower_pairs_removed':old_removed,'new_chassis5_tower_pairs':added,
 'excluded_proxy_pairs':len(excluded),'compiled_contact_pairs':m.npair,
 'limitations':'Unmerged convex chassis approximations, 95 near pairs excluded over sampled hip range. Other exclusions remain; not validated for training.'}
(p/'nomerge_collision_trial_build.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out,flush=True)
