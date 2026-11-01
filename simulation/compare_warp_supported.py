"""Compare stabilized CPU/GPU servo with temporary 35% simulated support."""
import json,math
from pathlib import Path
import numpy as np,mujoco,warp as wp,mujoco_warp as mjw
from supported_stepping_env import SupportedSteppingEnv
from stabilized_servo import StabilizedServo
from warp_servo import WarpServo
root=Path(__file__).parent;wp.config.kernel_cache_dir=str(root/'warp_cache');wp.init()
rows=[]
for seed in (9101,9102,9103):
 e=SupportedSteppingEnv(episode_seconds=4.,disturbance=.02,support_gain=1.3125,weight_support=.35)
 e.reset(seed=seed);e.controller=StabilizedServo(e.model,e.data)
 with wp.ScopedDevice('cuda:0'):
  gm=mjw.put_model(e.model);gd=mjw.put_data(e.model,e.data,nworld=1,nconmax=1024,njmax=4096)
  servo=WarpServo(gm,gd,e.controller.initial_positions,1)
  for k in range(200):
   phase=2*np.pi*k*.02;left=max(0,math.sin(phase));right=max(0,-math.sin(phase))
   a=np.array([-.5*left,.5*left,.5*right,-.5*right,.3*math.sin(phase)])
   servo.apply_support(e.body,1.3125,.35*sum(e.model.body_mass)*9.81)
   servo.set_actions(a[None,:]);servo.control_step();_,_,term,trunc,info=e.step(a)
   if k+1 in (1,25,50,100,200) or term or trunc:
    wp.synchronize();q=gd.qpos.numpy()[0];v=gd.qvel.numpy()[0]
    row=dict(seed=seed,seconds=(k+1)*.02,max_qpos_error=float(np.max(np.abs(q-e.data.qpos))),
     max_qvel_error=float(np.max(np.abs(v-e.data.qvel))),cpu_fell=bool(term),finite=bool(np.isfinite(q).all() and np.isfinite(v).all()))
    rows.append(row);print(json.dumps(row),flush=True)
   if term or trunc:break
 e.close()
(root/'warp_supported_comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
