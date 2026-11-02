"""Check reward preferences and short CPU/GPU correspondence without learning."""
from pathlib import Path
import json
import numpy as np
from sector_standing_reward import shaped_standing_reward as score
from sector_standing_env import SectorStandingEnv
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent
zero=np.zeros(5);still=np.zeros(3)
def single(u=1.,h=.28,w=still,a=zero,prev=zero,term=False):
    return float(score(u,h,.28,w,a,prev,term))
samples={'upright':single(), 'lean':single(u=.9), 'crouch':single(h=.24),
         'rotation':single(w=np.ones(3)), 'action':single(a=np.ones(5)),
         'terminated':single(term=True)}
if any(samples[key]>=samples['upright'] for key in ('lean','crouch','rotation','action','terminated')):
    raise RuntimeError('Reward preferences failed')
gpu=SectorWarpStandingVecEnv(nworld=4,episode_seconds=1.,reward_mode='shaped')
cpu=SectorStandingEnv(episode_seconds=1.,model_variant='implicit_stiff',
                      max_sampled_self_contact_n=20.,max_gear_residual_deg=.2,
                      max_pwm_delta_us_per_control=1.,reward_mode='shaped')
try:
    gpu.reset();cpu.reset(seed=1);rng=np.random.default_rng(9);differences=[]
    for _ in range(10):
        a=rng.uniform(-.2,.2,5)
        _,gr,gd,_=gpu.step(np.tile(a,(4,1)))
        _,cr,ct,cu,_=cpu.step(a)
        if gd.any() or ct or cu:raise RuntimeError('Unexpected termination')
        differences.append(abs(float(gr[0])-float(cr)))
    if max(differences)>1e-4:raise RuntimeError('CPU/GPU reward difference exceeded diagnostic tolerance')
    out={'synthetic_scores':samples,'short_cpu_gpu_max_abs_difference':max(differences),
         'frames':10,'worlds':4,'limitations':'Hypothetical reward and one short action sequence; no PPO learning, walking objective, model convergence or hardware validation.'}
    (p/'sector_shaped_reward_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)
finally:
    cpu.close();gpu.close()
