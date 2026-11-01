"""Verify scoped chassis0 replacement and preserved chassis5 collision pairs."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
data=json.loads((p/'chassis0_finer_proxy_range_audit.json').read_text(encoding='utf8'))
build=json.loads((p/'chassis0_collision_trial_build.json').read_text(encoding='utf8'))
near=data['near_pairs'];expected={tuple(r['geoms']) for r in near if r['min_mm']>.1}
excluded={tuple(r['geoms']) for r in near if r['min_mm']<=.1}
def pairs(filename):
 root=ET.parse(p/filename).getroot();return {(x.get('geom1'),x.get('geom2')) for x in root.find('contact').findall('pair')}
baseline=pairs('diagnostic_finer_collision_trial.xml');trial=pairs('diagnostic_chassis0_collision_trial.xml')
active={x for x in trial if (x[0] or '').startswith('nomerge_chassis0_')}
removed={tuple(x) for x in build['removed_pair_names']}
if active!=expected or active&excluded or baseline-removed != trial-active:
 raise RuntimeError('Scoped replacement did not preserve other collision pairs')
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_chassis0_collision_trial.xml'))
out={'valid':True,'restored_refined_pairs':len(active),'excluded_near_pairs':len(excluded),
 'removed_coarse_pairs':len(removed),'preserved_other_pairs':len(baseline-removed),'compiled_pairs':m.npair,
 'limitations':'Verifies selection against sampled proxy data and unchanged pair lists, not CAD clearance or full collision coverage.'}
(p/'chassis0_collision_selection_verification.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out,flush=True)
