"""Isolated no-contact, zero-gravity drive check; not a physical hardware test."""
from pathlib import Path
from dataclasses import replace
import json
import mujoco
import numpy as np
from theoretical_servo import ServoSpec
from candidate_standing_servo import CandidateStandingServo

p=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_post_shaft_fix_gear_phase_25.xml'))
m.opt.gravity[:]=0;m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
servo=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5)
names=['hypothetical_servo_shaft','L0_hip_candidate','L1_hip_candidate']
ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
initial=np.array([d.qpos[m.jnt_qposadr[j]] for j in ids]);max_res=0.;max_torque=0.
servo.set_pulse_us([1500,1500,1500,1500,1510])
for _ in range(round(4/m.opt.timestep)):
 servo.step()
 if not np.isfinite(d.qpos).all():raise RuntimeError('non-finite state')
 eq=(d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id<16)
 if np.any(eq):max_res=max(max_res,float(np.max(np.abs(d.efc_pos[eq]))))
 max_torque=max(max_torque,abs(float(d.ctrl[4])))
mujoco.mj_forward(m,d)
change=np.array([d.qpos[m.jnt_qposadr[j]] for j in ids])-initial
axis=np.array([np.dot(d.xaxis[j],d.xaxis[ids[0]]) for j in ids])
world=change*axis
out={'note':'Fixed 10 us simulated command in zero gravity with contact disabled; not a real-world actuator calibration.',
     'duration_s':float(d.time),'shaft_target_delta_deg':1.35,
     'common_axis_signed_joint_delta_deg':dict(zip(names,np.rad2deg(world).tolist())),
     'large_to_pinion_ratio_world':[float(world[i]/world[0]) for i in (1,2)],
     'max_leg_constraint_component_error_mm':max_res*1000,'max_shaft_torque_nm':max_torque}
(p/'gear_25_drive_isolated_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print('joint signed common-axis deltas deg',out['common_axis_signed_joint_delta_deg'])
print('ratios',out['large_to_pinion_ratio_world'],'max leg constraint mm',round(max_res*1000,5))
