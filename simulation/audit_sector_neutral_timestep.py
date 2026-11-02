"""Compare the assumed neutral servo model at three simulation time steps."""
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
for dt in (.00025,.000125,.0000625):
    model=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_stiff_trial.xml'))
    model.opt.timestep=dt
    data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
    servo=CandidateStandingServo(model,data,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
                                 kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=0.)
    hips=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')] for i in (0,1)]
    shaft=model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'hypothetical_servo_shaft')]
    initial=data.qpos.copy();peak_torque=np.zeros(5);peak_speed=np.zeros(5);stop=None;start=time.perf_counter()
    for _ in range(round(1./dt)):
        old=data.time;servo.step()
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.time<old+.5*dt:
            stop='nonfinite_or_reset';break
        if any(data.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL)):
            stop='capacity';break
        peak_torque=np.maximum(peak_torque,np.abs(data.ctrl));peak_speed=np.maximum(peak_speed,np.abs(data.actuator_velocity))
    mujoco.mj_forward(model,data)
    rows.append({'timestep_s':dt,'simulation_s':float(data.time),'stop_reason':stop,
                 'base_translation_m':data.qpos[:3].tolist(),
                 'base_quaternion_wxyz':data.qpos[3:7].tolist(),
                 'hip_delta_deg':np.rad2deg(data.qpos[hips]-initial[hips]).tolist(),
                 'shaft_delta_deg':float(np.rad2deg(data.qpos[shaft]-initial[shaft])),
                 'peak_torque_nm':peak_torque.tolist(),'peak_speed_rad_s':peak_speed.tolist(),
                 'final_servo_torque_nm':data.ctrl.tolist(),'wall_seconds':time.perf_counter()-start})
    print(json.dumps(rows[-1]),flush=True)
out={'results':rows,'limitations':'One second, neutral action and one set of unmeasured servo assumptions. Timestep comparisons do not validate CAD contacts or actual hardware.'}
(p/'sector_neutral_timestep_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
