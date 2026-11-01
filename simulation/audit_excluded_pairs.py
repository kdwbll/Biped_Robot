"""Measure previously excluded collision proxies; no learning or model edits."""
from pathlib import Path
from collections import Counter
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

p = Path(__file__).parent
def pairs(filename):
    return {tuple(sorted((x.get('geom1'), x.get('geom2'))))
            for x in ET.parse(p / filename).getroot().find('contact').findall('pair')}

excluded = sorted(pairs('biped_decomposed_chassis_trial.xml') - pairs('biped_filtered_collision_experimental.xml'))
m = mujoco.MjModel.from_xml_path(str(p / 'biped_filtered_collision_experimental.xml'))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
ids = [(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, a),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, b)) for a, b in excluded]
if any(min(pair) < 0 for pair in ids):
    raise RuntimeError('An excluded geom is missing from the candidate model')
rows = [{'geoms': list(pair), 'signed_distance_mm': float(mujoco.mj_geomDistance(m, d, *ix, 1.0, None) * 1000)}
        for pair, ix in zip(excluded, ids)]
out = {'model': 'biped_filtered_collision_experimental.xml', 'state': 'keyframe 0, before stepping',
       'method': 'MuJoCo signed distance between convex proxy geoms; negative is overlap',
       'limitations': 'Proxy geometry is approximate; overlap does not establish actual CAD interference. '
                      'Other states and physically acceptable contacts are not evaluated.',
       'excluded_count': len(rows), 'overlap_count': sum(r['signed_distance_mm'] < 0 for r in rows),
       'group_by_chassis_solid': dict(Counter(next(n for n in r['geoms'] if n.startswith('dc_chassis_')).rsplit('_', 1)[0] for r in rows)),
       'group_by_leg': dict(Counter(next(n for n in r['geoms'] if n.startswith('tower_')).split('_')[1] for r in rows)),
       'pairs': rows}
(p / 'excluded_pair_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('excluded:', len(rows), 'overlap:', out['overlap_count'], 'groups:', out['group_by_chassis_solid'], out['group_by_leg'])
print('deepest proxy distances mm:')
for r in sorted(rows, key=lambda x: x['signed_distance_mm'])[:8]:
    print(round(r['signed_distance_mm'], 3), *r['geoms'])
