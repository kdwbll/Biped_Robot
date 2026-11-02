"""Re-select partial chassis5 collisions using sampled full candidate hip range."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
audit=json.loads((p/'proxy_candidate_range_audit.json').read_text(encoding='utf8'))
prior=json.loads((p/'selective_collision_trial_build.json').read_text(encoding='utf8'))
old={tuple(x['geoms']) for x in prior['excluded_at_initial_pose']}
new={tuple(x['geoms']) for x in audit['near_pairs'] if x['min_mm']<=.1}
remove=new-old;restore=old-new
tree=ET.parse(p/'diagnostic_selective_collision_trial.xml');root=tree.getroot();contact=root.find('contact')
removed=[]
for pair in list(contact.findall('pair')):
 key=(pair.get('geom1'),pair.get('geom2'))
 if key in remove:contact.remove(pair);removed.append(key)
if set(removed)!=remove:raise RuntimeError('Previously active pairs not found')
for a,b in sorted(restore):ET.SubElement(contact,'pair',geom1=a,geom2=b,solref='.001 1',solimp='.99 .999 .001')
root.set('model','diagnostic_range_collision_trial')
model=p/'diagnostic_range_collision_trial.xml';tree.write(model,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(model));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
out={'source_model':'diagnostic_selective_collision_trial.xml','new_model':model.name,
 'sampled_hip_range_deg':audit['hypothetical_hip_range_deg'],'sample_count':audit['sample_count'],
 'sampled_clearance_margin_mm':.1,'previous_excluded':len(old),'new_excluded':len(new),
 'removed_previously_active_pairs':sorted(remove),'restored_previously_excluded_pairs':sorted(restore),
 'all_excluded_pairs':sorted(new),'compiled_pairs':m.npair,
 'limitations':'0.1 mm is an assumed proxy margin, not an exact CAD gap. Filtering only sampled rigid hip rotations; unresolved exclusions and other collision regions. Experimental; not for training.'}
(p/'range_collision_trial_build.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print({k:v for k,v in out.items() if k!='all_excluded_pairs'},flush=True)
