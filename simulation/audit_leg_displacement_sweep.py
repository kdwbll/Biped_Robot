"""Probe geometric foot displacement and contact under bounded open-loop actions."""
from pathlib import Path
import json
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent
rng=np.random.default_rng(314);actions=rng.choice([-1.,0.,1.],(16,5));actions[:,4]=0.;actions[0]=0.
env=SectorWarpStandingVecEnv(nworld=16,episode_seconds=4.,on_guard='terminate',pwm_slew_us=3.,action_amplitude_us=[200]*4+[20.])
try:
    env.reset();initial=env.gd.xpos.numpy()[:,env.foot_bodies,2].copy();max_delta=np.zeros((16,2));min_delta=np.zeros((16,2));guards=np.zeros(16,int);valid=np.zeros(16,int);air=np.zeros((16,2),int)
    for step in range(200):
        _,_,done,infos=env.step(np.zeros_like(actions) if step<25 else actions)
        dz=env.gd.xpos.numpy()[:,env.foot_bodies,2]-initial
        for i,info in enumerate(infos):
            if done[i] and info['guard_flags']:guards[i]+=1
            if guards[i]:continue
            max_delta[i]=np.maximum(max_delta[i],dz[i]);min_delta[i]=np.minimum(min_delta[i],dz[i]);valid[i]+=1
            if step>50:air[i]+=(np.asarray(info['sampled_sole_peak_normal_n'])<.5)
    rows=[{'action':actions[i].tolist(),'max_foot_origin_rise_mm':(max_delta[i]*1000).tolist(),
           'min_foot_origin_change_mm':(min_delta[i]*1000).tolist(),'guard_count':int(guards[i]),
           'valid_frames':int(valid[i]),'sampled_unload_frames':air[i].tolist()} for i in range(16)]
    out={'results':rows,'limitations':'One seed and 16 actions, fixed unmeasured mechanics; foot origin change is not clearance, contact is sampled, and no training or hardware.'}
    (p/'leg_displacement_sweep_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
