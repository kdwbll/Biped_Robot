"""Check finer trial collision selection against its saved sampled data."""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
audit=json.loads((p/'finer_proxy_range_audit.json').read_text(encoding='utf8'))
build=json.loads((p/'finer_collision_trial_build.json').read_text(encoding='utf8'))
expected={tuple(x['geoms']) for x in audit['near_pairs'] if x['min_mm']>.1}
excluded={tuple(x['geoms']) for x in audit['near_pairs'] if x['min_mm']<=.1}
root=ET.parse(p/'diagnostic_finer_collision_trial.xml').getroot()
active={(x.get('geom1'),x.get('geom2')) for x in root.find('contact').findall('pair') if (x.get('geom1') or '').startswith('nomerge_chassis5_')}
if active!=expected or active&excluded:raise RuntimeError('Incorrect collision selection')
if len(excluded)!=build['excluded_near_pairs']:raise RuntimeError('Incorrect exclusion count')
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_finer_collision_trial.xml'))
half_step=math.radians((audit['hip_range_deg'][1]-audit['hip_range_deg'][0])/(audit['sample_count']-1)/2)
motion_bound=2*audit['largest_tower_radius_m']*math.sin(half_step/2)*1000
if motion_bound>=2.:raise RuntimeError('Sampling motion bound exceeds retention distance')
out={'valid':True,'active_pairs':len(active),'excluded_pairs':len(excluded),'sample_count':audit['sample_count'],
 'max_between_sample_motion_bound_mm':motion_bound,'model_contact_pairs':m.npair,
 'limitation':'Motion bound assumes exact rigid rotation and accurate proxy distances. Selection matches saved sampled data; it is not a CAD/continuous collision proof.'}
(p/'finer_collision_selection_verification.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out,flush=True)
