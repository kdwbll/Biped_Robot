"""Check range-selected proxy pairs against the sampled range audit."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
dense='--dense' in sys.argv
audit=json.loads((p/('proxy_candidate_range_dense_audit.json' if dense else 'proxy_candidate_range_audit.json')).read_text(encoding='utf8'))
build=json.loads((p/'range_collision_trial_build.json').read_text(encoding='utf8'))
root=ET.parse(p/'diagnostic_range_collision_trial.xml').getroot()
active={(x.get('geom1'),x.get('geom2')) for x in root.find('contact').findall('pair')}
excluded={tuple(x) for x in build['all_excluded_pairs']}
unsafe={tuple(x['geoms']) for x in audit['near_pairs'] if x['min_mm']<=build['sampled_clearance_margin_mm']}
if unsafe!=excluded or active & excluded:raise RuntimeError('Collision selection does not match sampled thresholds')
for pair in build['restored_previously_excluded_pairs']:
 if tuple(pair) not in active:raise RuntimeError('Restored pair missing')
model=mujoco.MjModel.from_xml_path(str(p/'diagnostic_range_collision_trial.xml'))
out={'valid':True,'sample_count':audit['sample_count'],'compiled_pairs':model.npair,'excluded_pairs':len(excluded),
     'restored_pair_count':len(build['restored_previously_excluded_pairs']),
     'excluded_added_count':len(build['removed_previously_active_pairs']),
     'note':'Verifies consistency with previously computed discrete proxy distances; does not prove clearance between samples or collision coverage.'}
(p/('dense_range_collision_selection_verification.json' if dense else 'range_collision_selection_verification.json')).write_text(json.dumps(out,indent=2),encoding='utf8')
print(out,flush=True)
