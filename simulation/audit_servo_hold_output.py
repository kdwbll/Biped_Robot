"""Measure assumed simulated servo output during neutral standing. No training."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from stabilized_servo import StabilizedServo

root=Path(__file__).parent
rows=[]
for kp in (1., 4., 8.):
 for dt in (.0005, .00025):
  m=mujoco.MjModel.from_xml_path(str(root/'biped_fresh_mass_only_candidate.xml'))
  m.opt.timestep=dt
  d=mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m,d,0)
  mujoco.mj_forward(m,d)
  body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
  servo=StabilizedServo(m,d,replace(ServoSpec(),kp_nm_per_rad=kp))
  z0=float(d.xipos[body,2]); max_t=np.zeros(5); max_w=np.zeros(5); saturated=np.zeros(5,dtype=int)
  steps=round(2/dt)
  for i in range(steps):
   vel=d.actuator_velocity.copy()
   kd=np.array([servo.spec.kd_nm_per_rad_s]*4+[.01])
   raw=kp*(servo.target_positions-d.actuator_length)-kd*vel
   available=np.where(raw*vel>0,servo.spec.stall_torque_nm*np.maximum(0.,1.-np.abs(vel)/servo.spec.no_load_speed_rad_s),servo.spec.stall_torque_nm)
   saturated+=(np.abs(raw)>available+1e-9)
   servo.step()
   max_t=np.maximum(max_t,np.abs(d.ctrl)); max_w=np.maximum(max_w,np.abs(d.actuator_velocity))
   if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError((kp,dt,d.time))
  mujoco.mj_forward(m,d)
  row={'kp_nm_per_rad':kp,'timestep_s':dt,'duration_s':float(d.time),
       'height_loss_mm':float((z0-d.xipos[body,2])*1000),
       'max_abs_torque_nm_by_servo':max_t.tolist(),
       'max_abs_speed_rad_s_by_servo':max_w.tolist(),
       'final_torque_nm_by_servo':d.ctrl.tolist(),
       'final_angle_error_deg_by_servo':np.rad2deg(servo.target_positions-d.actuator_length).tolist(),
       'torque_limited_fraction_by_servo':(saturated/steps).tolist(),
       'stall_torque_nm_assumption':servo.spec.stall_torque_nm,
       'target_is_initial_actuator_position':True}
  rows.append(row)
  print('kp',kp,'dt',dt,'sag mm',round(row['height_loss_mm'],2),
        'max torque',np.round(max_t,3).tolist(),'limited fraction',np.round(saturated/steps,3).tolist(),
        'max speed',round(float(max(max_w)),2),flush=True)
(root/'servo_hold_output_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
