"""Validate Warp vector environment semantics and conservative aborts; no training."""
from pathlib import Path
import json,sys,time
import numpy as np
p=Path(__file__).parent
if (p/'warp_test_packages').exists():sys.path.insert(0,str(p/'warp_test_packages'))
import warp as wp
from sector_standing_env import SectorStandingEnv
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

env=SectorWarpStandingVecEnv(nworld=4,episode_seconds=1.);cpu=SectorStandingEnv(episode_seconds=1.,model_variant='implicit_stiff',max_sampled_self_contact_n=20.,max_gear_residual_deg=.2,max_pwm_delta_us_per_control=1.)
try:
 obs=env.reset();co,_=cpu.reset(seed=1)
 if obs.shape!=(4,16) or not np.isfinite(obs).all():raise RuntimeError('Invalid reset observations')
 try:env.step(np.full((4,5),np.nan));raise RuntimeError('NaN action accepted')
 except ValueError:pass
 errors=[];start=time.perf_counter()
 for step in range(4):
  actions=np.zeros((4,5),dtype=np.float32);actions[:,4]=.25
  obs,reward,done,infos=env.step(actions);co,cr,ct,cu,ci=cpu.step(actions[0])
  if done.any() or ct or cu or not np.isfinite(obs).all() or not np.isfinite(reward).all():raise RuntimeError('Unexpected ordinary termination')
  errors.append(float(np.max(np.abs(obs-co[None,:]))))
 elapsed=time.perf_counter()-start
 if max(errors)>.02:raise RuntimeError(('CPU/GPU observation divergence',max(errors)))
 if np.any(env.gd.xfrc_applied.numpy()):raise RuntimeError('Unexpected external virtual support')
 env.count[1]=env.horizon-1;other_before=env.gd.time.numpy().copy();obs,reward,done,infos=env.step(np.zeros((4,5),dtype=np.float32));other_after=env.gd.time.numpy()
 if done.tolist()!=[False,True,False,False] or not infos[1]['TimeLimit.truncated'] or 'terminal_observation' not in infos[1] or abs(float(other_after[1]))>1e-7 or not np.all(other_after[[0,2,3]]>other_before[[0,2,3]]):raise RuntimeError('Independent timeout reset failed')
 before=env.gd.time.numpy().copy();q=env.gd.qpos.numpy();q[2,env.hips[0]]+=np.deg2rad(4.6);wp.copy(env.gd.qpos,wp.array(q,dtype=wp.float32,device='cuda:0'))
 try:env.step(np.zeros((4,5),dtype=np.float32));raise AssertionError('Invalid world was stepped')
 except RuntimeError as e:
  if 'GPU diagnostic batch aborted' not in str(e):raise
 after=env.gd.time.numpy()
 if not np.array_equal(before,after):raise RuntimeError('Batch advanced after initial guard violation')
 try:env.step(np.zeros((4,5),dtype=np.float32));raise AssertionError('Aborted batch accepted step')
 except RuntimeError as e:
  if 'reset required' not in str(e):raise
 obs=env.reset()
 if not np.isfinite(obs).all():raise RuntimeError('Reset after abort failed')
 result={'worlds':4,'cpu_gpu_max_observation_difference':max(errors),'neutral_steps_each':4,'wall_seconds_including_cpu':elapsed,'masked_timeout_reset_passed':True,'injected_hip_violation_aborts_before_step':True,'aborted_batch_rejects_steps_until_reset':True,'nonfinite_action_rejected':True,'no_external_virtual_support':True,'limitations':'Short single-seed vector interface test, 4 worlds. Guard aborts entire batch and sampled contact checks can miss spikes; per-world friction randomization, extensive actions, learning, hardware and full parity remain unvalidated.'}
 (p/'sector_warp_vec_smoke_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2),flush=True)
finally:env.close()
