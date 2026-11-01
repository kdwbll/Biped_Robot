"""Experimental refinement of chassis-0 versus upper tower collision pairs."""
from pathlib import Path
import copy
import json
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
audit=json.loads((p/'chassis0_finer_proxy_range_audit.json').read_text(encoding='utf8'))
tree=ET.parse(p/'diagnostic_chassis0_finer_gap_only.xml');root=tree.getroot()
base=ET.parse(p/'diagnostic_finer_collision_trial.xml').getroot().find('contact')
old=root.find('contact');index=list(root).index(old);root.remove(old);contact=copy.deepcopy(base);root.insert(index,contact)
near=audit['near_pairs'];excluded={tuple(x['geoms']) for x in near if x['min_mm']<=.1}
active={tuple(x['geoms']) for x in near if x['min_mm']>.1}
keep=[];removed=[]
for pair in contact:
 a,b=pair.get('geom1'),pair.get('geom2')
 if (a.startswith('dc_chassis_0_') and (b.startswith('tower_') or b.startswith('refined_L'))) or (b.startswith('dc_chassis_0_') and (a.startswith('tower_') or a.startswith('refined_L'))):removed.append([a,b])
 else:keep.append(pair)
contact[:]=keep
for a,b in sorted(active):ET.SubElement(contact,'pair',geom1=a,geom2=b,solref='.001 1',solimp='.99 .999 .001')
root.set('model','diagnostic_chassis0_collision_trial');path=p/'diagnostic_chassis0_collision_trial.xml'
tree.write(path,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
out={'removed_coarse_pairs':len(removed),'removed_pair_names':removed,'restored_refined_pairs':len(active),
 'excluded_near_pairs':len(excluded),'compiled_contact_pairs':m.npair,
 'limitations':'Proxy pair filtering is restricted to sampled pure-hip range. Near pairs excluded; unverified parts, other motion and soft-limit overshoot remain. Not for training.'}
(p/'chassis0_collision_trial_build.json').write_text(json.dumps(out,indent=2),encoding='utf8');print({k:v for k,v in out.items() if k!='removed_pair_names'},flush=True)
