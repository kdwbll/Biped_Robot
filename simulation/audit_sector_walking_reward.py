"""Check a candidate walking reward and short CPU/GPU consistency without learning."""
from pathlib import Path
import json
import numpy as np
from sector_standing_reward import walking_reward
from sector_standing_env import SectorStandingEnv
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent
def value(velocity):
    return float(walking_reward(1.,.28,.28,np.zeros(3),np.zeros(5),np.zeros(5),False,velocity))
scores={'forward_target':value([.05,0]),'stationary':value([0,0]),'backward':value([-.05,0]),
        'sideways':value([0,.05]),'target_with_sideways':value([.05,.05])}
if not (scores['forward_target']>scores['stationary']>scores['backward'] and scores['forward_target']>scores['target_with_sideways']):
    raise RuntimeError('Velocity reward preference test failed')
gpu=SectorWarpStandingVecEnv(nworld=4,episode_seconds=1.,reward_mode='walking')
cpu=SectorStandingEnv(episode_seconds=1.,model_variant='implicit_stiff',max_sampled_self_contact_n=20.,
                      max_gear_residual_deg=.2,max_pwm_delta_us_per_control=1.,reward_mode='walking')
try:
    gpu.reset();cpu.reset(seed=1);rng=np.random.default_rng(10);errors=[]
    for _ in range(10):
        action=rng.uniform(-.2,.2,5)
        _,gr,gd,_=gpu.step(np.tile(action,(4,1)))
        _,cr,ct,cu,_=cpu.step(action)
        if gd.any() or ct or cu:raise RuntimeError('Unexpected termination')
        errors.append(abs(float(gr[0])-float(cr)))
    if max(errors)>1e-3:raise RuntimeError('Short CPU/GPU reward discrepancy exceeded test tolerance')
    out={'scores':scores,'short_cpu_gpu_max_abs_reward_difference':max(errors),'frames':10,'worlds':4,
         'limitations':'Candidate assumes world +X is forward, fixed 0.05 m/s target and uses simulation displacement. No footstep objective, full gait evaluation, optimization, or hardware validation.'}
    (p/'sector_walking_reward_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)
finally:cpu.close();gpu.close()
