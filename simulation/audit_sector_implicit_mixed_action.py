"""Exercise mixed smooth actions with an optional heuristic contact cutoff."""
from pathlib import Path
import json
import numpy as np
from sector_standing_env import SectorStandingEnv
p=Path(__file__).parent;rows=[]
for seed in (11,22,33):
 env=SectorStandingEnv(episode_seconds=2.,max_sampled_self_contact_n=20.,max_gear_residual_deg=.2,model_variant='implicit');env.reset(seed=seed);rng=np.random.default_rng(seed);action=np.zeros(5);peak=0.;gear=0.
 for step in range(100):
  if step%10==0:action=rng.uniform(-.3,.3,5)
  obs,reward,term,trunc,info=env.step(action);peak=max(peak,info['sampled_peak_self_contact_n']);gear=max(gear,info['max_gear_residual_deg'])
  if not np.isfinite(reward) or not np.isfinite(obs).all():raise RuntimeError('Invalid observation or reward')
  if term or trunc:break
 row={'seed':seed,'steps':step+1,'terminated':term,'truncated':trunc,'reason':info['reason'],'sampled_peak_n':peak,'max_gear_residual_deg':gear};rows.append(row);print(row,flush=True)
out={'guard_n':20.,'gear_guard_deg':.2,'variant':'implicit','results':rows,'limitations':'Only three short simulated trajectories, heuristic sampled-force guard, no training or hardware. Contact force threshold is not physically validated; actions are limited to +/-0.3.'}
(p/'sector_implicit_mixed_action_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
