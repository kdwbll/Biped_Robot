"""Compare excluded proxy pairs to source triangle meshes in the initial pose.

Read-only geometry diagnostic; triangle intersections are not solid CAD checks.
"""
from pathlib import Path
from collections import Counter
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import trimesh

p = Path(__file__).parent
model = mujoco.MjModel.from_xml_path(str(p / 'diagnostic_post_shaft_fix_gear_phase_25.xml'))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
source = trimesh.load_mesh(next((p / 'meshes_original_coords').glob('*16-40-03-958-1.STL')), process=True)
chassis = []
for part in source.split(only_watertight=False, repair=False):
    if part.is_watertight and len(part.faces) >= 20 and part.convex_hull.volume >= .0001:
        part.apply_scale(.001)
        chassis.append(part)
towers = []
for filename in ('single_leg_mesh_experiment.xml', 'second_leg_mesh_experiment.xml'):
    root = ET.parse(p / filename).getroot()
    parts = trimesh.load_mesh(p / root.find("./asset/mesh[@name='mesh0']").get('file'), process=True).split(only_watertight=False, repair=False)
    tower = max((x for x in parts if x.is_watertight and x.extents[2] > 100), key=lambda x: len(x.faces))
    tower.apply_scale(.001)
    towers.append(tower)

def pose(body):
    t = np.eye(4)
    t[:3, :3] = data.xmat[body].reshape(3, 3)
    t[:3, 3] = data.xpos[body]
    return t

excluded = json.loads((p / 'excluded_pair_audit.json').read_text(encoding='utf8'))['pairs']
groups = {}
for row in excluded:
    c = next(x for x in row['geoms'] if x.startswith('dc_chassis_'))
    t = next(x for x in row['geoms'] if x.startswith('tower_'))
    key = (int(c.split('_')[2]), int(t.split('_')[1][1:]))
    groups.setdefault(key, []).append((c, t))
results = []
for (idx, leg), pairs in sorted(groups.items()):
    ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pairs[0][0])
    ti = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pairs[0][1])
    if min(ci, ti) < 0:
        raise RuntimeError('Missing geometry')
    cm = trimesh.collision.CollisionManager()
    cm.add_object('chassis', chassis[idx], transform=pose(model.geom_bodyid[ci]))
    intersects, details = cm.in_collision_single(towers[leg], transform=pose(model.geom_bodyid[ti]), return_data=True)
    distance = float(cm.min_distance_single(towers[leg], transform=pose(model.geom_bodyid[ti])))
    results.append({'chassis_source_index': idx, 'leg': leg, 'excluded_pair_count': len(pairs),
                    'source_triangles_intersect': bool(intersects), 'source_contact_count': len(details),
                    'source_surface_distance_mm': distance * 1000,
                    'proxy_pairs': pairs})
out = {'model': 'diagnostic_post_shaft_fix_gear_phase_25.xml', 'pose': 'initial keyframe',
       'method': 'FCL source triangle mesh intersections, grouped by original chassis solid and tower',
       'limitations': 'An intersection anywhere in the two source meshes cannot identify whether each proxy pair is valid. No contact restoration, solid CAD check, or moving-state verification.',
       'excluded_proxy_pair_count': len(excluded), 'groups': results}
(p / 'pending_collision_source_audit.json').write_text(json.dumps(out, indent=2), encoding='utf8')
print('excluded proxy pairs', len(excluded), 'source groups', len(results))
for r in results:
    print('chassis', r['chassis_source_index'], 'leg', r['leg'], 'pairs', r['excluded_pair_count'], 'source triangles intersect', r['source_triangles_intersect'])
