"""Run several seeded random tilted resets for four seconds without learning."""
from pathlib import Path
import json
import sys
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p = Path(__file__).parent
tilt = float(sys.argv[1]) if len(sys.argv)>1 else 2.
env = SectorWarpStandingVecEnv(nworld=4, episode_seconds=4., on_guard='terminate', reset_tilt_deg=tilt)
try:
    env.seed(315)
    env.reset()
    initial_tilts = env.reset_tilts.tolist()
    guard_events = []
    falls = []
    peak_force = 0.
    truncations = 0
    for step in range(200):
        obs,reward,done,infos = env.step(np.zeros((4,5)))
        if not np.isfinite(obs).all() or not np.isfinite(reward).all(): raise RuntimeError('Nonfinite result')
        for i, info in enumerate(infos):
            peak_force = max(peak_force, info['sampled_peak_self_contact_n'])
            if done[i]:
                if info['guard_flags']:guard_events.append({'step':step,'world':i,'flags':info['guard_flags'],'peak_force_n':info['sampled_peak_self_contact_n'],'peak_pair':info['sampled_peak_self_contact_pair']})
                elif info['TimeLimit.truncated']:truncations += 1
                else:falls.append([step,i])
    out={'reset_tilt_limit_deg':tilt,'initial_tilts_deg':initial_tilts,'control_frames':200,'guard_events':guard_events,
         'falls':falls,'time_limit_truncations':truncations,'peak_sampled_self_contact_n':peak_force,
         'limitations':'Four worlds, one seed and zero actions for four seconds; fixed hypothetical friction, sampled contacts, no learning or hardware validation.'}
    (p/f'sector_random_tilt_episode_{tilt:g}deg_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)
finally:
    env.close()
