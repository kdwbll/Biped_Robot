"""Compare hypothetical hip-servo gains against a small ground command; no training."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
cases=[(4.,1.5,.00025),(8.,6.,.00025),(12.,10.,.00025),(16.,15.,.00025),
       (8.,6.,.000125),(12.,10.,.000125)]
out=[]
for kp,ki,dt in cases:
 m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_post_shaft_fix_gear_phase_25.xml'));m.opt.timestep=dt
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
 servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
                              kp_by_servo=[4,4,4,4,kp],ki_by_servo=[1.5,1.5,1.5,1.5,ki])
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
 shaft=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')
 ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
 q0=float(d.qpos[m.jnt_qposadr[shaft]]);base=d.xipos[body].copy();max_speed=0.;max_torque=0.;max_tilt=0.;peak_contact=0.;invalid=None
 for i in range(round(6/dt)):
  if d.time>=2.:servo.set_pulse_us([1500,1500,1500,1500,1510])
  previous=float(d.time);servo.step()
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<previous+dt*.5:
   invalid=f'nonfinite or automatic reset at {previous:.6f} seconds';break
  max_speed=max(max_speed,abs(float(d.actuator_velocity[4])));max_torque=max(max_torque,abs(float(d.ctrl[4])))
  max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(d.xmat[body].reshape(3,3)[2,2],-1,1)))))
  if i%20==0:
   for ci,c in enumerate(d.contact[:d.ncon]):
    if ground in (c.geom1,c.geom2):continue
    f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);peak_contact=max(peak_contact,float(f[0]))
 mujoco.mj_forward(m,d)
 row={'kp_hip_nm_per_rad':kp,'ki_hip_nm_per_rad_s':ki,'dt_s':dt,'valid':invalid is None,'invalid_reason':invalid,
      'duration_s':float(d.time),'actual_shaft_delta_deg':float(np.degrees(d.qpos[m.jnt_qposadr[shaft]]-q0)),
      'shaft_target_error_deg':float(np.degrees(servo.target_positions[4]-d.actuator_length[4])),
      'max_hip_torque_nm':max_torque,'max_hip_speed_rad_s':max_speed,'max_tilt_deg':max_tilt,
      'horizontal_drift_mm':float(np.linalg.norm((d.xipos[body]-base)[:2])*1000),
      'sampled_max_self_contact_n':peak_contact,
      'notes':'Simulated 10 us command at t=2s, default ground contacts. Gains are unmeasured hypotheses.'}
 out.append(row)
 print('kp',kp,'ki',ki,'dt',dt,'valid',row['valid'],'shaft deg',round(row['actual_shaft_delta_deg'],3),
       'error deg',round(row['shaft_target_error_deg'],3),'peak torque',round(max_torque,3),
       'speed',round(max_speed,2),'self N',round(peak_contact,2),flush=True)
(p/'hip_servo_gains_diagnostic.json').write_text(json.dumps(out,indent=2),encoding='utf8')
