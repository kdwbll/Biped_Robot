"""Construct a scoped collision trial from finer sampled proxy distances."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
audit=json.loads((p/'finer_proxy_range_audit.json').read_text(encoding='utf8'))
if audit['sample_count']!=201 or audit['hip_range_deg']!=[-5,5]:raise RuntimeError('Unexpected sample set')
rows=audit['near_pairs'];excluded={tuple(x['geoms']) for x in rows if x['min_mm']<=.1}
retained={tuple(x['geoms']) for x in rows if x['min_mm']>.1}
tree=ET.parse(p/'diagnostic_finer_full_gap_only.xml');root=tree.getroot();contact=root.find('contact');keep=[];removed=0
for pair in contact:
 a,b=pair.get('geom1'),pair.get('geom2')
 if (a.startswith('dc_chassis_5_') and (b.startswith('tower_') or b.startswith('refined_L'))) or (b.startswith('dc_chassis_5_') and (a.startswith('tower_') or a.startswith('refined_L'))):removed+=1
 else:keep.append(pair)
contact[:]=keep
for a,b in sorted(retained):ET.SubElement(contact,'pair',geom1=a,geom2=b,solref='.001 1',solimp='.99 .999 .001')
root.set('model','diagnostic_finer_collision_trial')
path=p/'diagnostic_finer_collision_trial.xml';tree.write(path,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
out={'near_pairs':len(rows),'retained_chassis_tower_pairs':len(retained),'excluded_near_pairs':len(excluded),
 'removed_old_pairs':removed,'model_contact_pairs':m.npair,
 'limitations':'Only assumed rigid hip range was sampled. Near pairs remain excluded; no coverage outside range, under constraint drift or other movement. Experimental and not for training.'}
(p/'finer_collision_trial_build.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out,flush=True)
