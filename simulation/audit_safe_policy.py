import json,numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from safer_clocked_env import SaferClockedEnv
p=Path(__file__).parent;model=PPO.load(str(p/'iteration_safe_32768.zip'),device='cpu');rows=[]
for mode in ('neutral','policy'):
 for seed in (7001,7002,7003,7004,7005):
  e=SaferClockedEnv(episode_seconds=4,disturbance=.02);o,_=e.reset(seed=seed);x=e.data.xipos[e.body,0];peak=np.zeros(2);swings=0;tilt=0
  for i in range(200):
   a=np.zeros(5,dtype=np.float32) if mode=='neutral' else model.predict(o,deterministic=True)[0];o,r,term,tr,info=e.step(a);peak=np.maximum(peak,info.get('foot_clearance_m',[0,0]));swings+=int(info.get('valid_swing',False));tilt=max(tilt,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
   if term or tr:break
  row={'mode':mode,'seed':seed,'seconds':(i+1)*.02,'distance_mm':1000*(e.data.xipos[e.body,0]-x),'foot_peak_mm':(peak*1000).tolist(),'supported_swing_frames':swings,'tilt_max_deg':tilt,'fall':bool(term)};rows.append(row);print(row,flush=True)
(p/'iteration_safe_32768_detailed_eval.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
