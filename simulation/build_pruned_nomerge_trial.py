"""Experimental sampled-range broadphase pruning of unmerged proxy pairs."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco

p=Path(__file__).parent
near=json.loads((p/'nomerge_proxy_pruning_audit.json').read_text(encoding='utf8'))
blocked=json.loads((p/'nomerge_proxy_range_audit.json').read_text(encoding='utf8'))
if near['sample_count']!=201 or near['selection_threshold_mm']!=2.:raise RuntimeError('Unexpected pruning audit')
excluded={tuple(x['geoms']) for x in blocked['near_pairs']}
within={tuple(x['geoms']) for x in near['near_pairs']}
if not excluded <= within:raise RuntimeError('Near-pair audit did not include all exclusions')
tree=ET.parse(p/'diagnostic_nomerge_collision_trial.xml');root=tree.getroot();contact=root.find('contact');removed=0;keep=[]
for pair in contact:
 key=(pair.get('geom1'),pair.get('geom2'))
 if (key[0] or '').startswith('nomerge_chassis5_') and key not in within:
  removed+=1
 else:keep.append(pair)
contact[:]=keep
root.set('model','diagnostic_pruned_nomerge_trial')
path=p/'diagnostic_pruned_nomerge_trial.xml';tree.write(path,encoding='utf8',xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
out={'sampled_hip_range_deg':[-5,5],'samples':201,'retention_distance_mm':2.,
 'near_pairs':len(within),'unresolved_excluded_pairs':len(excluded),'active_chassis_tower_pairs':len(within-excluded),
 'removed_far_pairs':removed,'compiled_pairs':m.npair,
 'limitations':'Experimental culling only for assumed pure hip rotation in +/-5 degrees, based on sampled proxy distance. No guarantee outside that range, under geometric/constraint errors, or for unresolved pairs; not for training.'}
(p/'pruned_nomerge_trial_build.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out,flush=True)
