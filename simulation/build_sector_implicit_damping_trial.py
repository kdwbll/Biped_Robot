"""Build simulation-only implicit damping and hypothetical bearing-friction candidate."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import mujoco
p=Path(__file__).parent;t=ET.parse(p/'diagnostic_sector_axial_contact_trial.xml');root=t.getroot();changes=[]
for leg in (0,1):
 j=root.find(f".//joint[@name='L{leg}_hip_candidate']")
 if j is None or j.get('frictionloss')!='0':raise RuntimeError('Unexpected hip configuration')
 j.set('frictionloss','0.1');changes.append({'joint':j.get('name'),'frictionloss_nm':.1})
shaft=root.find(".//joint[@name='hypothetical_servo_shaft']")
if shaft is None or shaft.get('damping')!='0.001':raise RuntimeError('Unexpected shaft damping')
shaft.set('damping','0.301');changes.append({'joint':'hypothetical_servo_shaft','damping_nm_s_per_rad':.301})
root.set('model','diagnostic_sector_implicit_damping_trial');out=p/'diagnostic_sector_implicit_damping_trial.xml';t.write(out,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(out));report={'model':out.name,'changes':changes,'contact_pairs':int(m.npair),'required_external_servo_hip_damping':0.,'limitations':'Friction and damping are unmeasured hypotheses; requires simulation controller configured with zero explicit hip derivative to avoid double-counting. Some collision pairs absent; XML has no runtime range guard. Not hardware validated.'}
(p/'sector_implicit_damping_trial_build.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
