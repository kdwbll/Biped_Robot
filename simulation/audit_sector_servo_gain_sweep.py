"""Compare hypothetical shaft servo gains, without selecting real hardware settings."""
from pathlib import Path
from dataclasses import replace
import json
import time
import mujoco
import numpy as np
from candidate_standing_servo import CandidateStandingServo
from theoretical_servo import ServoSpec

p=Path(__file__).parent
rows=[]
for kp,ki in ((8.,6.),(12.,3.),(16.,3.)):
    m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_stiff_trial.xml'));m.opt.timestep=.00025
    hips=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')] for i in (0,1)]
    for i in (0,1):
        j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate');m.dof_frictionloss[m.jnt_dofadr[j]]=.05
    d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
    s=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
                             kp_by_servo=[4,4,4,4,kp],ki_by_servo=[1.5,1.5,1.5,1.5,ki],hip_damping=0.)
    q0=d.actuator_length[4];h0=d.qpos[hips].copy();max_speed=0.;max_torque=0.;max_hip=0.;stop=None;start=time.perf_counter()
    for step in range(round(3./m.opt.timestep)):
        if step==round(.5/m.opt.timestep):s.set_pulse_us([1500,1500,1500,1500,1480])
        before=d.time;s.step()
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<before+.5*m.opt.timestep:
            stop='nonfinite_or_reset';break
        if any(d.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL)):
            stop='capacity';break
        max_speed=max(max_speed,abs(float(d.actuator_velocity[4])));max_torque=max(max_torque,abs(float(d.ctrl[4])))
        max_hip=max(max_hip,float(np.max(np.abs(np.rad2deg(d.qpos[hips]-h0)))))
        if max_hip>4.5:stop='hip_range';break
    mujoco.mj_forward(m,d)
    rows.append({'kp_nm_per_rad':kp,'ki_nm_per_rad_s':ki,'simulation_s':float(d.time),'stop_reason':stop,
                 'shaft_delta_deg':float(np.rad2deg(d.actuator_length[4]-q0)),
                 'shaft_error_deg':float(np.rad2deg(s.target_positions[4]-d.actuator_length[4])),
                 'peak_shaft_speed_rad_s':max_speed,'peak_shaft_torque_nm':max_torque,'peak_hip_change_deg':max_hip,
                 'wall_seconds':time.perf_counter()-start})
    print(json.dumps(rows[-1]),flush=True)
out={'results':rows,'limitations':'Three assumed gain settings, one timestep, friction value and command; no measured servo behavior, comprehensive contact check, training or hardware validation.'}
(p/'sector_servo_gain_sweep_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
