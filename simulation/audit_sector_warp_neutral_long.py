"""Measure neutral standing in GPU simulation, without learning or rendering."""
from pathlib import Path
import json
import time
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

HERE = Path(__file__).parent
env = SectorWarpStandingVecEnv(nworld=16, episode_seconds=10.0,
                               on_guard='terminate')
result = {'worlds': 16, 'horizon_seconds': 10.0, 'control_frames': 0,
          'guard_terminations': 0, 'fall_terminations': 0,
          'time_limit_truncations': 0, 'max_self_contact_n': 0.0,
          'minimum_upright': 1.0, 'min_height_m': None,
          'max_height_m': None, 'batch_abort': None}
try:
    env.reset()
    start = time.perf_counter()
    for _ in range(500):
        try:
            obs, reward, done, infos = env.step(np.zeros((16, 5)))
        except RuntimeError as exc:
            result['batch_abort'] = str(exc)
            break
        if not np.isfinite(obs).all() or not np.isfinite(reward).all():
            raise RuntimeError('Nonfinite result')
        result['control_frames'] += 1
        for i, info in enumerate(infos):
            result['max_self_contact_n'] = max(result['max_self_contact_n'], info['sampled_peak_self_contact_n'])
            result['minimum_upright'] = min(result['minimum_upright'], info['upright'])
            h = info['height']
            result['min_height_m'] = h if result['min_height_m'] is None else min(result['min_height_m'], h)
            result['max_height_m'] = h if result['max_height_m'] is None else max(result['max_height_m'], h)
            if done[i]:
                if info['guard_flags']: result['guard_terminations'] += 1
                elif info['TimeLimit.truncated']: result['time_limit_truncations'] += 1
                else: result['fall_terminations'] += 1
    result['wall_seconds'] = time.perf_counter() - start
    result['world_steps_per_second'] = 16 * result['control_frames'] / result['wall_seconds']
    result['limitations'] = ('Neutral action only, identical initial conditions, fixed hypothetical friction; '
                             'sampled contact force can miss spikes. No learning or hardware verification.')
    (HERE / 'sector_warp_neutral_long_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
finally:
    env.close()
