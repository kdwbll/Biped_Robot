"""Check optional per-control-step PWM slew limiting without training."""
from pathlib import Path
import json
import numpy as np
from sector_standing_env import SectorStandingEnv
p=Path(__file__).parent;env=SectorStandingEnv(episode_seconds=2,model_variant='implicit_stiff',max_pwm_delta_us_per_control=1.,max_gear_residual_deg=.2,max_sampled_self_contact_n=20.);env.reset(seed=1);prev=np.full(5,1500.)
for i in range(20):
 _,_,term,trunc,info=env.step(np.ones(5));now=np.array(info['applied_pwm_us'])
 if term or trunc or np.max(np.abs(now-prev))>1+1e-10:raise RuntimeError(('slew forward failed',i,info))
 prev=now
if not np.allclose(prev,1520):raise RuntimeError(('unexpected pulse after 20 steps',prev))
_,_,term,trunc,info=env.step(-np.ones(5));reverse=np.array(info['applied_pwm_us'])
if term or trunc or not np.allclose(reverse,1519):raise RuntimeError('slew reverse failed')
env.reset(seed=1);_,_,_,_,info=env.step(np.zeros(5))
if not np.allclose(info['applied_pwm_us'],1500):raise RuntimeError('reset pulse failed')
for bad in (0.,-1.,float('nan')):
 try:SectorStandingEnv(max_pwm_delta_us_per_control=bad);raise RuntimeError('invalid limit accepted')
 except ValueError:pass
out={'forward_steps':20,'slew_limit_us_per_control':1.,'all_pulses_after_forward_us':prev.tolist(),'all_pulses_after_reverse_us':reverse.tolist(),'reset_and_invalid_limit_checks_passed':True,'limitations':'1 us per 20 ms is an arbitrary test limit, not a measured hardware parameter. Does not validate all actions or full training.'}
(p/'sector_pwm_slew_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
