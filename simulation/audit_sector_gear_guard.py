"""Check optional gear residual guard for both simulation variants."""
from pathlib import Path
import json
import numpy as np
from sector_standing_env import SectorStandingEnv
p=Path(__file__).parent;rows=[]
for variant in ('baseline','implicit'):
 env=SectorStandingEnv(episode_seconds=1,model_variant=variant,max_gear_residual_deg=.2);env.reset(seed=1)
 env.data.qpos[env.hip_qadr[0]]+=np.deg2rad(.3);t=float(env.data.time)
 _,_,terminated,truncated,info=env.step(np.zeros(5))
 if not terminated or truncated or info['reason']!='gear_guard' or env.data.time!=t or info['max_gear_residual_deg']<.299:raise RuntimeError((variant,info))
 env.reset(seed=1);_,_,term,_,info2=env.step(np.zeros(5))
 if term:raise RuntimeError(('unexpected normal stop',variant,info2))
 rows.append({'variant':variant,'injected_error_deg':.3,'threshold_deg':.2,'stopped_before_step':True,'normal_step_accepted':True})
for invalid in (0.,-1.,float('nan')):
 try:SectorStandingEnv(max_gear_residual_deg=invalid);raise RuntimeError('invalid guard accepted')
 except ValueError:pass
out={'results':rows,'invalid_thresholds_rejected':True,'limitations':'0.2 degrees is a test setting, not a validated real gear tolerance. Guard only covers one ideal relation, can overshoot within one physics step and exists only in this environment.'}
(p/'sector_gear_guard_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
