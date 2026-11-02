"""Verify seeded friction randomization and short guarded mixed trajectories."""
from pathlib import Path
import json
import numpy as np
from sector_standing_env import SectorStandingEnv
p=Path(__file__).parent;env=SectorStandingEnv(episode_seconds=2,model_variant='implicit_stiff',friction_range=(.05,.2),max_sampled_self_contact_n=20.,max_gear_residual_deg=.2)
_,a=env.reset(seed=41);_,b=env.reset(seed=41)
if a['hypothetical_hip_friction_nm']!=b['hypothetical_hip_friction_nm']:raise RuntimeError('Seeded friction not reproducible')
_,c=env.reset(seed=42)
if c['hypothetical_hip_friction_nm']==a['hypothetical_hip_friction_nm']:raise RuntimeError('Friction did not vary')
for bad in ((-.1,.2),(.2,.1),(float('nan'),.2)):
 try:SectorStandingEnv(model_variant='implicit_stiff',friction_range=bad);raise RuntimeError('Bad friction accepted')
 except ValueError:pass
rows=[]
for seed in (11,22,33):
 _,initial=env.reset(seed=seed);rng=np.random.default_rng(seed);action=np.zeros(5);peak=0.;gear=0.
 for step in range(100):
  if step%10==0:action=rng.uniform(-.3,.3,5)
  obs,reward,term,trunc,info=env.step(action);peak=max(peak,info['sampled_peak_self_contact_n']);gear=max(gear,info['max_gear_residual_deg'])
  if not np.isfinite(obs).all() or not np.isfinite(reward):raise RuntimeError('nonfinite output')
  if term or trunc:break
 row={'seed':seed,'assumed_friction_nm':initial['hypothetical_hip_friction_nm'],'steps':step+1,'terminated':term,'truncated':trunc,'reason':info['reason'],'sampled_peak_n':peak,'max_gear_residual_deg':gear};rows.append(row);print(row,flush=True)
out={'friction_range_nm':[.05,.2],'seed_reproducible':True,'invalid_ranges_rejected':True,'results':rows,'limitations':'Only three short trajectories and arbitrary unmeasured friction range. Guards are simulation heuristics; this is not training or hardware validation.'}
(p/'sector_randomized_friction_smoke_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
