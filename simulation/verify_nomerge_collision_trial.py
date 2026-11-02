"""Verify that sampled near pairs are the only unmerged pairs omitted."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
audit=json.loads((p/'nomerge_proxy_range_audit.json').read_text(encoding='utf8'))
build=json.loads((p/'nomerge_collision_trial_build.json').read_text(encoding='utf8'))
root=ET.parse(p/'diagnostic_nomerge_collision_trial.xml').getroot()
active={(x.get('geom1'),x.get('geom2')) for x in root.find('contact').findall('pair')
        if (x.get('geom1') or '').startswith('nomerge_chassis5_')}
excluded={tuple(r['geoms']) for r in audit['near_pairs']}
if active & excluded or len(active)!=audit['pair_count']-len(excluded):raise RuntimeError('Incorrect proxy selection')
if len(excluded)!=build['excluded_proxy_pairs']:raise RuntimeError('Incorrect excluded count')
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_nomerge_collision_trial.xml'))
out={'valid':True,'total_pairs':audit['pair_count'],'active_pairs':len(active),'excluded_pairs':len(excluded),
     'compiled_contact_pairs':m.npair,'sample_count':audit['sample_count'],
     'limitations':'Only verifies model selection against the sampled proxy data; no continuous clearance or CAD proof.'}
(p/'nomerge_collision_selection_verification.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(out,flush=True)
