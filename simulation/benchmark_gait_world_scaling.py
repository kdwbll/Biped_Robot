"""Measure short GPU diagnostic throughput at a selected world count; no learning."""
from pathlib import Path
import json,sys,time
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent;n=int(sys.argv[1]) if len(sys.argv)>1 else 32
env=SectorWarpStandingVecEnv(nworld=n,episode_seconds=2.,on_guard='terminate',reset_tilt_deg=1.,reward_mode='shaped',pwm_slew_us=2.,action_amplitude_us=[150]*4+[20.])
try:
 env.seed(117);env.reset();rng=np.random.default_rng(117);count=guards=falls=0;start=time.perf_counter()
 for step in range(50):
  if step%10==0:action=rng.uniform(-.2,.2,(n,5))
  obs,reward,done,infos=env.step(action)
  if not np.isfinite(obs).all() or not np.isfinite(reward).all():raise RuntimeError('Nonfinite result')
  count+=1
  for end,info in zip(done,infos):
   if end and info['guard_flags']:guards+=1
   elif end and not info['TimeLimit.truncated']:falls+=1
 elapsed=time.perf_counter()-start
 out={'worlds':n,'frames':count,'world_steps_per_second':n*count/elapsed,'wall_s':elapsed,'guards':guards,'falls':falls,
      'limitations':'One second of small random actions, one seed, no rendering or training; guard and sampled collisions remain approximate.'}
 (p/f'gait_scaling_{n}_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
