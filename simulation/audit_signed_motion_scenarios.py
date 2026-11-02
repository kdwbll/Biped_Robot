"""Evaluate small bidirectional servo commands in a fixed model, without learning."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent
rows=[]
for axis, delta in [(4,-10),(4,10),(0,-5),(0,5),(1,-5),(1,5),(2,-5),(2,5),(3,-5),(3,5)]:
    m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_post_shaft_fix_gear_phase_25.xml'))
    d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
    servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
         kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6])
    b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_free')
    ground=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ground')
    initial=d.xipos[b].copy();start=servo.initial_positions.copy()
    peak_force=0.;max_tilt=0.;max_torque=0.;valid=True
    for step in range(round(4/m.opt.timestep)):
        if d.time>=1.:
            pulse=[1500]*5;pulse[axis]+=delta;servo.set_pulse_us(pulse)
        old=float(d.time);servo.step()
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<old+m.opt.timestep*.5:
            valid=False;break
        max_torque=max(max_torque,abs(float(d.ctrl[axis])))
        if step%20==0:
            max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(d.xmat[b].reshape(3,3)[2,2],-1,1)))))
            for i,c in enumerate(d.contact[:d.ncon]):
                if ground in (c.geom1,c.geom2):continue
                f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);peak_force=max(peak_force,float(f[0]))
    mujoco.mj_forward(m,d)
    row={'servo_index_zero_based':axis,'pulse_delta_us':delta,'valid':valid,'simulated_seconds':float(d.time),
         'actual_delta_deg':float(np.degrees(d.actuator_length[axis]-start[axis])),
         'target_error_deg':float(np.degrees(servo.target_positions[axis]-d.actuator_length[axis])),
         'height_change_mm':float((d.xipos[b,2]-initial[2])*1000),
         'max_sampled_tilt_deg':max_tilt,'max_commanded_torque_nm':max_torque,
         'max_sampled_self_contact_n':peak_force}
    rows.append(row);print(row,flush=True)
(p/'signed_motion_scenarios_audit.json').write_text(json.dumps({'results':rows,
 'limitations':'Hypothetical simulation gains, unmeasured 270 degree servo mapping and incomplete collisions; force and tilt sampled every 20 steps; not a physical range test.'},indent=2),encoding='utf8')
