"""Check seeded GPU pose resets and floor clearance without training."""
from pathlib import Path
import json
import mujoco
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p = Path(__file__).parent
env = SectorWarpStandingVecEnv(nworld=4, episode_seconds=.2, on_guard='terminate', reset_tilt_deg=2.)
try:
    env.seed(314)
    env.reset()
    first = env.gd.qpos.numpy().copy()
    tilts = env.reset_tilts.copy()
    clearances = []
    for q in first:
        data = mujoco.MjData(env.m)
        data.qpos[:] = q
        mujoco.mj_forward(env.m, data)
        distances = [float(c.dist) for c in data.contact[:data.ncon] if env.ground in (c.geom1,c.geom2)]
        clearances.append(min(distances,default=0.))
    env.seed(314)
    env.reset()
    reproducible = np.array_equal(first, env.gd.qpos.numpy()) and np.array_equal(tilts, env.reset_tilts)
    before = env.gd.qpos.numpy().copy()
    env._reset_mask(np.array([False,True,False,False]))
    after = env.gd.qpos.numpy()
    isolated = np.array_equal(before[[0,2,3]],after[[0,2,3]])
    changed = not np.array_equal(before[1],after[1])
    obs,reward,done,info = env.step(np.zeros((4,5)))
    if not reproducible or not isolated or not changed or min(clearances)<-2e-6 or not np.isfinite(obs).all():
        raise RuntimeError('Seed, reset isolation, initial clearance or finite observation check failed')
    out = {'seed':314, 'tilts_deg':tilts.tolist(), 'minimum_floor_distances_m':clearances,
           'seed_reproducible':reproducible, 'other_world_qpos_unchanged_at_reset':isolated,
           'reset_world_changed':changed, 'first_frame_done':done.tolist(),
           'guard_flags':[x['guard_flags'] for x in info],
           'limitations':'Four worlds, one seed, one step and one isolated reset; no long-run isolation, training or hardware validation.'}
    (p/'sector_random_tilt_reset_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)
finally:
    env.close()
