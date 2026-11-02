"""Check optional sampled-force cutoff using fixed diagnostic actions, no training."""
from pathlib import Path
import json
import numpy as np
from sector_standing_env import SectorStandingEnv
p=Path(__file__).parent;rows=[]
for limit,sign in ((20.,-1.),(20.,1.),(100.,-1.)):
 env=SectorStandingEnv(episode_seconds=4.,max_sampled_self_contact_n=limit);env.reset(seed=1);action=np.array([0,0,0,0,sign])
 for step in range(200):
  _,reward,term,trunc,info=env.step(action)
  if term or trunc:break
 row={'guard_limit_n':limit,'action_sign':sign,'steps':step+1,'terminated':term,'truncated':trunc,'reason':info['reason'],'simulation_time':info['simulation_time'],'last_sampled_self_contact_n':info['sampled_peak_self_contact_n'],'pair':info['sampled_peak_pair']}
 rows.append(row);print(row,flush=True)
 if limit==20 and (not term or info['reason']!='contact_guard' or info['sampled_peak_self_contact_n']<=limit):raise RuntimeError('Expected force guard cutoff')
 if limit==100 and (term or not trunc):raise RuntimeError('Unexpected termination for high threshold')
out={'results':rows,'limitations':'20 N and 100 N are arbitrary test thresholds, not physical safety limits. Forces sampled every 2.5 ms may miss peaks. No guarantee for other actions; only this Python environment enforces the optional guard.'}
(p/'sector_sampled_force_guard_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
