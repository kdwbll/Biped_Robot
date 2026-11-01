"""Ablation of suspected self-collision artifacts; diagnostic, not a replacement model."""
from pathlib import Path
from dataclasses import replace
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
tree=ET.parse(p/'biped_standing_gear_candidate.xml');root=tree.getroot();pairs=root.find('contact');removed=0
for pair in list(pairs.findall('pair')):
 if 'ground' not in (pair.get('geom1'),pair.get('geom2')):
  pairs.remove(pair);removed+=1
root.set('model','diagnostic_standing_gear_ground_only');ET.indent(root)
floor=p/'diagnostic_standing_gear_ground_only.xml';tree.write(floor,encoding='utf8',xml_declaration=True)
rows=[]
for path in [p/'biped_standing_gear_candidate.xml',floor]:
 m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
 servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');z0=d.xipos[body,2];max_tilt=0.;max_speed=0.
 for _ in range(round(6/m.opt.timestep)):
  servo.step();max_speed=max(max_speed,float(np.max(np.abs(d.actuator_velocity))))
  max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(d.xmat[body].reshape(3,3)[2,2],-1,1)))))
  if not np.isfinite(d.qpos).all():raise RuntimeError(path.name)
 mujoco.mj_forward(m,d);force_max_ground=force_max_self=0.
 for i,c in enumerate(d.contact[:d.ncon]):
  f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
  names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)]
  if 'ground' in names:force_max_ground=max(force_max_ground,float(f[0]))
  else:force_max_self=max(force_max_self,float(f[0]))
 row={'model':path.name,'removed_self_collision_pairs':0 if path!=floor else removed,
      'duration_s':float(d.time),'height_loss_mm':float((z0-d.xipos[body,2])*1000),
      'final_servo_error_deg':np.rad2deg(servo.target_positions-d.actuator_length).tolist(),
      'final_servo_torque_nm':d.ctrl.tolist(),'max_tilt_deg':max_tilt,'max_servo_speed_rad_s':max_speed,
      'final_max_single_ground_contact_force_n':force_max_ground,'final_max_single_self_contact_force_n':force_max_self}
 rows.append(row)
 print(path.name,'sag mm',round(row['height_loss_mm'],3),'fifth error deg',round(row['final_servo_error_deg'][4],3),
       'force ground/self N',round(force_max_ground,2),round(force_max_self,2),'tilt deg',round(max_tilt,2),flush=True)
(p/'collision_force_ablation.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
