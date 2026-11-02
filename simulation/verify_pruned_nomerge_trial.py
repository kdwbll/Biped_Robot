"""Check sampled broadphase list and pointwise clearance error budget."""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

p=Path(__file__).parent
near=json.loads((p/'nomerge_proxy_pruning_audit.json').read_text(encoding='utf8'))
blocked=json.loads((p/'nomerge_proxy_range_audit.json').read_text(encoding='utf8'))
build=json.loads((p/'pruned_nomerge_trial_build.json').read_text(encoding='utf8'))
within={tuple(x['geoms']) for x in near['near_pairs']};excluded={tuple(x['geoms']) for x in blocked['near_pairs']}
root=ET.parse(p/'diagnostic_pruned_nomerge_trial.xml').getroot()
active={(x.get('geom1'),x.get('geom2')) for x in root.find('contact').findall('pair') if (x.get('geom1') or '').startswith('nomerge_chassis5_')}
if active!=within-excluded:raise RuntimeError('Pruned collision list does not match sampled near pairs')
m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_pruned_nomerge_trial.xml'));d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
radii=[]
for leg in (0,1):
 joint=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{leg}_hip_candidate')
 ids=[i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(f'highres_L{leg}_')]
 radii.append(max(float(np.linalg.norm(d.geom_xpos[i]-d.xanchor[joint])+m.geom_rbound[i]) for i in ids))
half_step_rad=math.radians((near['hip_range_deg'][1]-near['hip_range_deg'][0])/(near['sample_count']-1)/2)
bound_mm=2*max(radii)*math.sin(half_step_rad/2)*1000
if bound_mm>=build['retention_distance_mm']:raise RuntimeError('Sample spacing budget exceeds retention distance')
out={'valid':True,'active_pairs':len(active),'blocked_pairs':len(excluded),'sample_count':near['sample_count'],
 'largest_tower_proxy_sphere_radius_m':max(radii),'max_between_sample_motion_bound_mm':bound_mm,
 'retention_distance_mm':build['retention_distance_mm'],
 'limitations':'Geometric motion bound assumes pure rigid rotation around the sampled hip axes, fixed relative torso and no numerical/geometry error. Does not validate soft-limit overshoot, other motion, CAD surfaces, or 95 excluded pairs.'}
(p/'pruned_nomerge_selection_verification.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(out,flush=True)
