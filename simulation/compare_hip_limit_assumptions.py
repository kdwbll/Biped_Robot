"""Contrast 0.2 degrees and 0.2 radians; neither is a confirmed hardware limit."""
import json
import mujoco,numpy as np
from stabilized_servo import StabilizedServo
from supported_stepping_env import SupportedSteppingEnv

rows=[]
for limit in ('file_degrees','hypothesis_0p2rad'):
 for command in ('neutral','hip_sine'):
  e=SupportedSteppingEnv(episode_seconds=4.,disturbance=0.,support_gain=0.,weight_support=0.)
  e.reset(seed=1);ids=[mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate') for i in (0,1)]
  if limit=='hypothesis_0p2rad':
   for j in ids:e.model.jnt_range[j]=[-.2,.2]
   mujoco.mj_forward(e.model,e.data)
  e.controller=StabilizedServo(e.model,e.data);addresses=[e.model.jnt_qposadr[j] for j in ids]
  initial=e.data.qpos[addresses].copy();peak=np.zeros(2);violation=np.zeros(2);shaftpeak=0.;tilt=0.;start=e.data.xipos[e.body].copy()
  for k in range(200):
   action=np.zeros(5)
   if command=='hip_sine':action[-1]=.5*np.sin(2*np.pi*k*.02)
   _,_,term,trunc,info=e.step(action);peak=np.maximum(peak,np.abs(e.data.qpos[addresses]-initial))
   for leg,j in enumerate(ids):
    q=e.data.qpos[addresses[leg]];lo,hi=e.model.jnt_range[j];violation[leg]=max(violation[leg],lo-q,q-hi,0.)
   shaftpeak=max(shaftpeak,abs(float(e.data.qvel[-1])));tilt=max(tilt,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
   if term or trunc:break
  row=dict(assumption=limit,command=command,range_deg=np.degrees(e.model.jnt_range[ids[0]]).tolist(),duration_s=(k+1)*.02,
   max_hip_degrees=np.degrees(peak).tolist(),max_soft_limit_violation_degrees=np.degrees(violation).tolist(),sampled_shaft_speed_rad_s=shaftpeak,max_tilt_deg=tilt,
   body_height_change_mm=float((e.data.xipos[e.body,2]-start[2])*1000),terminated=bool(term))
  rows.append(row);e.close();print(json.dumps(row),flush=True)
open('hip_limit_assumptions_comparison.json','w',encoding='utf8').write(json.dumps(rows,indent=2))
