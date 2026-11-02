"""Exercise candidate standing interface with seeded random actions, without PPO updates."""
from pathlib import Path
import json
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent
env=SectorWarpStandingVecEnv(nworld=4,episode_seconds=2.,on_guard='terminate',reset_tilt_deg=1.,reward_mode='shaped')
rows=[]
try:
    for seed in (42,43,44):
        env.seed(seed);env.reset();rng=np.random.default_rng(seed);events=[];returns=np.zeros(4);max_force=0.;truncations=0
        for step in range(100):
            if step%10==0:action=rng.uniform(-.2,.2,(4,5))
            obs,reward,done,infos=env.step(action);returns+=reward
            if not np.isfinite(obs).all() or not np.isfinite(reward).all():raise RuntimeError('Nonfinite simulation result')
            for i,info in enumerate(infos):
                max_force=max(max_force,info['sampled_peak_self_contact_n'])
                if done[i]:
                    if info['TimeLimit.truncated']:truncations+=1
                    else:events.append({'frame':step,'world':i,'guard_flags':info['guard_flags'],
                                        'force_n':info['sampled_peak_self_contact_n'],
                                        'pair':info['sampled_peak_self_contact_pair']})
        rows.append({'seed':seed,'returns':returns.tolist(),'early_terminations':events,
                     'time_limit_truncations':truncations,'peak_sampled_self_contact_n':max_force})
    out={'results':rows,'limitations':'Four worlds per seed, small random open-loop actions for two seconds. No optimizer or learned policy; guard thresholds and dynamics remain approximate.'}
    (p/'sector_shaped_random_actions_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)
finally:env.close()
