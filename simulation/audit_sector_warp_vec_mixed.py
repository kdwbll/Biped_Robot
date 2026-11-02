"""Run short low-amplitude mixed GPU vector actions, without training."""
from pathlib import Path
import json,sys,time,numpy as np
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv
env=SectorWarpStandingVecEnv(nworld=4,episode_seconds=2.,contact_limit_n=20.,gear_limit_deg=.2,pwm_slew_us=1.);rng=np.random.default_rng(13);out={}
try:
 obs=env.reset();action=np.zeros((4,5));maxforce=0.;start=time.perf_counter();error=None;completed=0
 for step in range(100):
  if step%10==0:action=rng.uniform(-.3,.3,(4,5))
  try:obs,reward,done,infos=env.step(action)
  except RuntimeError as exc:error=str(exc);break
  if not np.isfinite(obs).all() or not np.isfinite(reward).all():raise RuntimeError('Nonfinite observation')
  maxforce=max(maxforce,max(x['sampled_peak_self_contact_n'] for x in infos));completed+=1
  if done.any() and step!=99:raise RuntimeError(('Unexpected early episode done',step,done.tolist()))
 elapsed=time.perf_counter()-start
 out={'worlds':4,'seed':13,'control_frames':completed,'total_world_steps':4*completed,'wall_seconds':elapsed,'world_steps_per_second':4*completed/elapsed,'all_horizon_resets':bool(completed==100 and done.all()),'max_sampled_self_contact_n':maxforce,'batch_abort':error,'limitations':'One 2 s low-amplitude action schedule, fixed unmeasured friction and heuristic guards; host sync and no renderer. No PPO training. Not evidence of hardware safety or broad walking robustness.'}
 (p/'sector_warp_vec_mixed_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
