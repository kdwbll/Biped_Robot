"""Verify slip changes reward only for loaded feet, without training."""
from pathlib import Path
import json
import mujoco
import numpy as np
from sector_standing_env import SectorStandingEnv
from sector_foot_slip import sole_sample_slip

p=Path(__file__).parent
env=SectorStandingEnv(episode_seconds=1.,model_variant='implicit_stiff',reward_mode='walking',slip_weight=.1)
try:
    values=[]
    for weight in (0.,.1):
        env.slip_weight=weight;env.reset(seed=7)
        env.data.qvel[0]=.1;mujoco.mj_forward(env.model,env.data)
        _,reward,term,trunc,info=env.step(np.zeros(5))
        if term or trunc:raise RuntimeError('Unexpected termination')
        values.append((float(reward),info['sampled_contact_slip_mps']))
    expected=.02*.1*sum(values[1][1]);difference=values[0][0]-values[1][0]
    if expected<=0 or abs(expected-difference)>1e-9:raise RuntimeError('Slip penalty did not match sampled velocity')
    cvel=np.zeros((1,1,6));cvel[0,0,3]=1.
    _,air=sole_sample_slip([[0,1]],[0],[[0,0,0]],[0.],cvel,np.zeros((1,1,3)),0,[1,2],[0,0],[0,0],1)
    if np.any(air):raise RuntimeError('Unloaded foot was penalized')
    out={'without_penalty_reward':values[0][0],'with_penalty_reward':values[1][0],
         'sampled_slip_mps':values[1][1],'observed_penalty':difference,'expected_penalty':expected,
         'unloaded_foot_slip_mps':air.tolist(),
         'limitations':'One simulated initial translation and one control step. Contact samples every 2.5 ms, with hypothetical 0.5 N load threshold and 0.1 weight; no policy training or hardware validation.'}
    (p/'sector_slip_penalty_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
finally:env.close()
