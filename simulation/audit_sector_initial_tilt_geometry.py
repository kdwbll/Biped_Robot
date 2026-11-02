"""Measure contact distances before stepping after a rigid initial tilt."""
from pathlib import Path
import json
import mujoco
import numpy as np

p = Path(__file__).parent
m = mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_stiff_trial.xml'))
ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'ground')
pair = {'dc_chassis_3_2', 'refined_L0_tower_15'}
rows = []
for angle in (0., -2., 2.):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    a = np.deg2rad(angle)/2
    w, x, y, z = d.qpos[3:7].copy()
    c, s = np.cos(a), np.sin(a)
    d.qpos[3:7] = [c*w-s*x, c*x+s*w, c*y-s*z, c*z+s*y]
    mujoco.mj_forward(m, d)
    floor = []
    internal = []
    for contact in d.contact[:d.ncon]:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, j) for j in (contact.geom1, contact.geom2)]
        if ground in (contact.geom1, contact.geom2): floor.append(float(contact.dist))
        if set(names) == pair: internal.append(float(contact.dist))
    rows.append({'roll_change_deg':angle, 'ground_contact_count':len(floor),
                 'minimum_ground_distance_m':min(floor) if floor else None,
                 'pair_contact_count':len(internal),
                 'minimum_pair_distance_m':min(internal) if internal else None})
out = {'results':rows, 'limitations':'Rigid root rotation around the free-joint origin, before simulation step; contact proxy distances do not prove CAD interference or real-world loads.'}
(p/'sector_initial_tilt_geometry_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
