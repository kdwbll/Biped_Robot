"""Test neutral controller response to small simulated initial tilts; no learning."""
from pathlib import Path
import json
import time
import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

HERE = Path(__file__).parent
cases = [(axis, angle) for axis in ('roll', 'pitch') for angle in (-2., -1., 1., 2.)]
env = SectorWarpStandingVecEnv(nworld=len(cases), episode_seconds=4.0, on_guard='terminate')
try:
    env.reset()
    qpos = env.gd.qpos.numpy()
    # floating_base is a free joint; multiply a world-frame tilt by the initial quaternion.
    lift = []
    for i, (axis, angle) in enumerate(cases):
        a = np.deg2rad(angle) / 2
        w, x, y, z = qpos[i, 3:7]
        c, s = np.cos(a), np.sin(a)
        qpos[i, 3:7] = ([c*w-s*x, c*x+s*w, c*y-s*z, c*z+s*y] if axis == 'roll'
                         else [c*w-s*y, c*x+s*z, c*y+s*w, c*z-s*x])
        probe = mujoco.MjData(env.m)
        probe.qpos[:] = qpos[i]
        mujoco.mj_forward(env.m, probe)
        floor_distances = [contact.dist for contact in probe.contact[:probe.ncon]
                           if env.ground in (contact.geom1, contact.geom2)]
        correction = max(0., -min(floor_distances, default=0.))
        qpos[i, 2] += correction
        lift.append(float(correction))
    wp.copy(env.gd.qpos, wp.array(qpos, dtype=wp.float32, device='cuda:0'))
    mjw.forward(env.gm, env.gd)
    wp.synchronize()
    first = env._observation()
    rows = [{'axis': axis, 'initial_tilt_deg': angle, 'ground_clearance_correction_m': lift[i], 'initial_gravity': first[i, :3].tolist(),
             'max_tilt_deg': 0., 'final_tilt_deg': None, 'first_terminal_s': None,
             'guard_flags': 0, 'max_sampled_self_contact_n': 0., 'peak_pair': None} for i, (axis, angle) in enumerate(cases)]
    start = time.perf_counter()
    for step in range(200):
        obs, reward, done, infos = env.step(np.zeros((len(cases), 5)))
        for i, row in enumerate(rows):
            if row['first_terminal_s'] is not None: continue
            tilt = float(np.rad2deg(np.arccos(np.clip(infos[i]['upright'], -1, 1))))
            row['final_tilt_deg'] = tilt
            row['max_tilt_deg'] = max(row['max_tilt_deg'], tilt)
            if infos[i]['sampled_peak_self_contact_n'] > row['max_sampled_self_contact_n']:
                row['max_sampled_self_contact_n'] = infos[i]['sampled_peak_self_contact_n']
                row['peak_pair'] = infos[i]['sampled_peak_self_contact_pair']
            if done[i] and not infos[i]['TimeLimit.truncated']:
                row['first_terminal_s'] = (step + 1) * .02
                row['guard_flags'] = infos[i]['guard_flags']
    out = {'results': rows, 'wall_seconds': time.perf_counter()-start,
           'limitations': 'Eight small initial roll/pitch perturbations at zero commanded action, raised as needed to remove initial floor penetration; fixed assumed friction and heuristic sampled guards; no active learned balancing, training or hardware validation.'}
    (HERE/'sector_warp_tilt_response_audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)
finally:
    env.close()
