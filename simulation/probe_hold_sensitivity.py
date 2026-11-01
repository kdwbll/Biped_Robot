"""Check neutral sag against assumed proportional gain and collision proxies."""
from pathlib import Path
from dataclasses import replace
import json,xml.etree.ElementTree as ET
import mujoco,numpy as np
from theoretical_servo import ServoSpec
from stabilized_servo import StabilizedServo
from supported_stepping_env import SupportedSteppingEnv

ROOT=Path(__file__).parent
tree=ET.parse(ROOT/'biped_filtered_collision_experimental.xml');contact=tree.getroot().find('contact')
for p in list(contact):
 if p.tag=='pair' and 'ground' not in (p.get('geom1'),p.get('geom2')):contact.remove(p)
path=ROOT/'diagnostic_floor_only.xml';tree.write(path,encoding='utf8',xml_declaration=True)

rows=[]
for model_name in ('biped_filtered_collision_experimental.xml','diagnostic_floor_only.xml'):
 for kp in (1.,2.,4.,8.):
  e=SupportedSteppingEnv(episode_seconds=2.,disturbance=0.,support_gain=0.,weight_support=0.)
  if model_name!='biped_filtered_collision_experimental.xml':
   e.model=mujoco.MjModel.from_xml_path(str(ROOT/model_name));e.data=mujoco.MjData(e.model)
   e.body=mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
   e.foot_geom=[mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('L0_visual6','L1_visual6')]
  e.reset(seed=1);e.controller=StabilizedServo(e.model,e.data,replace(ServoSpec(),kp_nm_per_rad=kp))
  z0=float(e.data.xipos[e.body,2]);peak_v=0.;peak_torque=0.;peak_tilt=0.
  for k in range(100):
   _,_,term,trunc,info=e.step(np.zeros(5));peak_v=max(peak_v,float(np.max(np.abs(e.data.qvel))))
   peak_torque=max(peak_torque,float(np.max(np.abs(e.data.ctrl))));peak_tilt=max(peak_tilt,float(np.degrees(np.arccos(np.clip(info['upright'],-1,1)))))
   if term or trunc:break
  row=dict(model=model_name,kp_nm_per_rad=kp,seconds=(k+1)*.02,initial_height_m=z0,
    final_height_m=float(e.data.xipos[e.body,2]),height_loss_mm=float((z0-e.data.xipos[e.body,2])*1000),
    sampled_max_qvel_rad_s=peak_v,sampled_max_torque_nm=peak_torque,max_tilt_deg=peak_tilt,terminated=bool(term))
  rows.append(row);e.close();print(json.dumps(row),flush=True)
(ROOT/'hold_sensitivity_diagnostic.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
