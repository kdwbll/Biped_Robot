"""Run deterministic behavior checks for the sector standing environment, not training."""
from pathlib import Path
import json
import numpy as np
from sector_standing_env import SectorStandingEnv
p=Path(__file__).parent;env=SectorStandingEnv(episode_seconds=1.0);obs,info=env.reset(seed=7)
assert env.observation_space.contains(obs)
try:env.step([0,0,np.nan,0,0]);raise AssertionError('NaN action accepted')
except ValueError:pass
neutral=[]
for _ in range(50):
 obs,reward,term,trunc,info=env.step(np.zeros(5));assert env.observation_space.contains(obs) and np.isfinite(reward);assert np.isfinite(info['sampled_peak_self_contact_n']) and np.isfinite(info['max_gear_residual_deg']);neutral.append([obs.tolist(),reward])
 if term or trunc:break
assert trunc and not term and info['reason'] is None
try:env.step(np.zeros(5));raise AssertionError('Step after done accepted')
except RuntimeError:pass
env.reset(seed=7);repeat=[]
for _ in range(50):
 obs,reward,term,trunc,info=env.step(np.zeros(5));repeat.append([obs.tolist(),reward])
 if term or trunc:break
assert np.allclose(np.array([r[0] for r in neutral]),np.array([r[0] for r in repeat]),rtol=0,atol=1e-10)
env.reset(seed=7);env.data.qpos[env.hip_qadr[0]]+=np.deg2rad(4.6);t=float(env.data.time)
_,_,terminated,_,info=env.step(np.zeros(5));assert terminated and info['reason']=='hip_range' and env.data.time==t
env.reset(seed=7);env.data.qvel[0]=np.nan;t=float(env.data.time)
_,_,terminated,_,info=env.step(np.zeros(5));assert terminated and info['reason']=='nonfinite' and env.data.time==t
env.reset(seed=9);rng=np.random.default_rng(9);reason=None
for step in range(50):
 obs,reward,term,trunc,info=env.step(rng.uniform(-1,1,5));assert env.observation_space.contains(obs) and np.isfinite(reward)
 if term or trunc:reason=info['reason'];break
result={'neutral_1s_completed':len(neutral)==50,'neutral_reproducible':True,'contact_and_gear_metrics_finite':True,'invalid_action_rejected':True,'post_terminal_step_rejected':True,'hip_violation_stops_before_step':True,'nonfinite_stops_before_step':True,'random_actions_steps':step+1,'random_termination_reason':reason,'limitations':'Short deterministic smoke test; not training, not full collision validation or real-hardware safety. Guard exists only within this environment.'}
(p/'sector_standing_env_smoke_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2),flush=True)
