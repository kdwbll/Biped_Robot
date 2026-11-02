"""Exercise all five bounded actions separately, without training or hardware."""
from pathlib import Path
import json
import numpy as np
from sector_standing_env import SectorStandingEnv
p=Path(__file__).parent;rows=[]
for axis in range(5):
 for sign in (-1.,1.):
  env=SectorStandingEnv(episode_seconds=2);env.reset(seed=1);action=np.zeros(5);action[axis]=sign;peak=0.;gear=0.;maxhip=0.;reason=None
  for step in range(100):
   obs,reward,term,trunc,info=env.step(action);peak=max(peak,info['sampled_peak_self_contact_n']);gear=max(gear,info['max_gear_residual_deg']);maxhip=max(maxhip,info.get('max_hip_deg',0.));reason=info.get('reason')
   if term or trunc:break
  row={'action_axis':axis,'action_sign':sign,'steps':step+1,'terminated':term,'truncated':trunc,'reason':reason,'max_sampled_self_contact_n':peak,'max_gear_residual_deg':gear,'max_hip_change_deg':maxhip}
  rows.append(row);print(row,flush=True)
out={'results':rows,'limitations':'10 independent two-second single-axis actions, approximate and partial contacts, sampled forces, not training or hardware; combined and longer actions remain untested.'}
(p/'sector_standing_env_single_axis_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
