"""Score independent unsupported rollouts, including separate foot lift events."""
import json,sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stepping_env import SteppingEnv

def audit(path,seeds=(9001,9002,9003,9004,9005)):
 model=PPO.load(str(path),device='cpu');rows=[]
 for seed in seeds:
  e=SteppingEnv(episode_seconds=4.,disturbance=.02);obs,_=e.reset(seed=seed)
  x0=float(e.data.xipos[e.body,0]);last=[False,False];events=[0,0];frames=[0,0];peak=[0.,0.];order=[];tilt=0.
  for k in range(200):
   action,_=model.predict(obs,deterministic=True);obs,_,term,trunc,info=e.step(action)
   contacts=info['foot_contacts'];clear=info['foot_clearance_m']
   for leg in (0,1):
    peak[leg]=max(peak[leg],clear[leg]);swing=bool(clear[leg]>.003 and contacts[1-leg])
    frames[leg]+=int(swing)
    if swing and not last[leg]:events[leg]+=1;order.append(leg)
    last[leg]=swing
   tilt=max(tilt,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
   if term or trunc:break
  rows.append(dict(seed=seed,seconds=(k+1)*.02,forward_m=float(e.data.xipos[e.body,0]-x0),
                   fell=bool(term),max_tilt_deg=tilt,peak_clearance_m=peak,
                   supported_lift_frames=frames,supported_lift_events=events,event_order=order))
  e.close()
 return dict(policy=str(path),support='none',episodes=rows)

if __name__=='__main__':
 path=Path(sys.argv[1]);out=path.with_suffix('.walking_audit.json')
 result=audit(path);out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)
