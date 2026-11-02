"""Restore 30 proxy pairs with positive bounded clearances in an independent trial."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import mujoco
p=Path(__file__).parent
baseline='diagnostic_separate_sole_contact_trial.xml'
audit=json.loads((p/'near_collision_gap_refinement_audit.json').read_text(encoding='utf8'))
rows=audit['results']
if len(rows)!=30 or any(x['certified_proxy_gap_lower_bound_mm']<=.02 for x in rows):raise RuntimeError('Insufficient clearance margin')
tree=ET.parse(p/baseline);root=tree.getroot();contact=root.find('contact')
existing={frozenset((x.get('geom1'),x.get('geom2'))) for x in contact.findall('pair')}
for row in rows:
 a,b=row['geoms']
 if frozenset((a,b)) in existing:raise RuntimeError('Collision pair already active')
 ET.SubElement(contact,'pair',geom1=a,geom2=b,solref='.001 1',solimp='.99 .999 .001')
root.set('model','diagnostic_restored_near_pairs_trial');out=p/'diagnostic_restored_near_pairs_trial.xml';tree.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
report={'source':baseline,'model':out.name,'restored_pair_count':len(rows),'remaining_chassis0_near_pairs':790,'compiled_pair_count':int(m.npair),'minimum_bounded_proxy_gap_mm':min(x['certified_proxy_gap_lower_bound_mm'] for x in rows),'limitations':'Bounds cover assumed +/-5 degree hip sweep only and ideal rigid proxy geometry, not physical tolerances or other excluded pairs. XML does not enforce hip range. Diagnostic only, not for training or hardware.'}
(p/'restored_near_pairs_build.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
