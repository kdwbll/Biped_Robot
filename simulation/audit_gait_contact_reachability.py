"""Probe whether allowed small actions can unload either foot; no learning."""
from pathlib import Path
import json
import sys
import numpy as np
from sector_warp_standing_vec_env import SectorWarpStandingVecEnv

p=Path(__file__).parent
actions=np.array([[0,0,0,0,0],[-1,0,0,0,0],[1,0,0,0,0],[0,-1,0,0,0],
                  [0,1,0,0,0],[0,0,-1,0,0],[0,0,1,0,0],[0,0,0,-1,0],
                  [0,0,0,1,0],[-1,1,0,0,0],[1,-1,0,0,0],[0,0,-1,1,0],
                  [0,0,1,-1,0],[-1,-1,1,1,0],[1,1,-1,-1,0],[0,0,0,0,1]],dtype=float)
amplitude=float(sys.argv[1]) if len(sys.argv)>1 else 30.
env=SectorWarpStandingVecEnv(nworld=len(actions),episode_seconds=3.,on_guard='terminate',reset_tilt_deg=0.,pwm_slew_us=2.,action_amplitude_us=[amplitude]*4+[20.])
try:
    env.reset();unloaded=np.zeros((len(actions),2),int);peak=np.zeros((len(actions),2));guards=np.zeros(len(actions),int);fall=np.zeros(len(actions),int)
    for step in range(150):
        _,_,done,infos=env.step(np.zeros_like(actions) if step<25 else actions)
        for i,info in enumerate(infos):
            forces=np.asarray(info['sampled_sole_peak_normal_n']);peak[i]=np.maximum(peak[i],forces)
            if step>=100 and not info['guard_flags']:unloaded[i]+=forces<.5
            if done[i] and info['guard_flags']:guards[i]+=1
            elif done[i] and not info['TimeLimit.truncated']:fall[i]+=1
    out={'results':[{'action':actions[i].tolist(),'unloaded_fraction_last_second':(unloaded[i]/50).tolist(),
                     'peak_sampled_contact_per_point_n':peak[i].tolist(),'guard_episodes':int(guards[i]),'fall_episodes':int(fall[i])} for i in range(len(actions))],
         'limitations':'Open-loop three-second probes, 0.5 N single-contact peak heuristic; no ground truth foot height, learning, walking or hardware validation.'}
    out['leg_action_amplitude_us']=amplitude
    (p/f'gait_contact_reachability_{amplitude:g}us_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2),flush=True)
finally:env.close()
