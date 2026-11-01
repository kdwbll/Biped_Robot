"""Isolate ground-friction influence on a small hip command; diagnostic only."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent;out=[]
for friction_factor in (1.,.1,0.):
 m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_post_shaft_fix_gear_phase_25.xml'))
 ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
 if friction_factor != 1.:
  for i in range(m.npair):
   if ground in (m.pair_geom1[i],m.pair_geom2[i]):m.pair_friction[i,:]*=friction_factor
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
 servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free');base=d.xipos[body].copy()
 shaft=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft');initial=d.qpos[m.jnt_qposadr[shaft]]
 samples=[];peak_torque=0.;peak_speed=0.;invalid_reason=None
 for _ in range(round(8/m.opt.timestep)):
  if d.time>=2.:servo.set_pulse_us([1500,1500,1500,1500,1510])
  old_time=float(d.time)
  servo.step();peak_torque=max(peak_torque,abs(float(d.ctrl[4])));peak_speed=max(peak_speed,abs(float(d.actuator_velocity[4])))
  if not np.isfinite(d.qpos).all():raise RuntimeError('nonfinite')
  if d.time < old_time+m.opt.timestep*.5 or np.linalg.norm(d.qpos[:3])>100:
   invalid_reason=f'numerical instability or automatic reset at {old_time:.6f} s';break
  if abs(d.time-4)<m.opt.timestep/2 or abs(d.time-8)<m.opt.timestep/2:
   mujoco.mj_forward(m,d)
   samples.append({'time_s':float(d.time),'actual_shaft_delta_deg':float(np.degrees(d.qpos[m.jnt_qposadr[shaft]]-initial)),
                   'shaft_target_error_deg':float(np.degrees(servo.target_positions[4]-d.actuator_length[4])),
                   'shaft_torque_nm':float(d.ctrl[4]),'horizontal_drift_mm':float(np.linalg.norm((d.xipos[body]-base)[:2])*1000)})
 row={'ground_friction_factor':friction_factor,'valid':invalid_reason is None,'invalid_reason':invalid_reason,
      'command':'simulated 1510 us starting at 2 s, model maps to +1.35 deg',
      'samples':samples,'peak_shaft_torque_nm':peak_torque,'peak_shaft_velocity_rad_s':peak_speed,
      'note':'Changing ground friction is an ablation, not a validated floor model.'}
 out.append(row)
 print('friction factor',friction_factor,'valid',row['valid'],'reason',invalid_reason,'samples',[(round(x['time_s']),round(x['actual_shaft_delta_deg'],3),
       round(x['shaft_target_error_deg'],3),round(x['horizontal_drift_mm'],2)) for x in samples],flush=True)
(p/'hip_ground_friction_diagnostic.json').write_text(json.dumps(out,indent=2),encoding='utf8')
