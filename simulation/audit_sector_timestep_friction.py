"""Compare step response across timestep and hypothetical hip friction."""
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
for friction in (0.,.05):
    for dt in (.00025,.000125):
        m=mujoco.MjModel.from_xml_path(str(p/'diagnostic_sector_implicit_stiff_trial.xml'));m.opt.timestep=dt
        for i in (0,1):
            j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'L{i}_hip_candidate')
            m.dof_frictionloss[m.jnt_dofadr[j]]=friction
        d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
        s=CandidateStandingServo(m,d,replace(ServoSpec(),kp_nm_per_rad=4.),ki=1.5,
                                 kp_by_servo=[4,4,4,4,8],ki_by_servo=[1.5,1.5,1.5,1.5,6],hip_damping=0.)
        initial=d.actuator_length.copy();max_speed=0.;stop=None;start=time.perf_counter()
        for step in range(round(2./dt)):
            if step==round(.5/dt):s.set_pulse_us([1500,1500,1500,1500,1480])
            before=d.time;s.step()
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time<before+.5*dt:
                stop='nonfinite_or_reset';break
            if any(d.warning[int(w)].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,mujoco.mjtWarning.mjWARN_CNSTRFULL)):
                stop='capacity';break
            max_speed=max(max_speed,abs(float(d.actuator_velocity[4])))
        mujoco.mj_forward(m,d)
        rows.append({'frictionloss_nm_per_hip':friction,'timestep_s':dt,'simulation_s':float(d.time),
                     'stop_reason':stop,'shaft_delta_deg':float(np.rad2deg(d.actuator_length[4]-initial[4])),
                     'shaft_error_deg':float(np.rad2deg(s.target_positions[4]-d.actuator_length[4])),
                     'peak_shaft_speed_rad_s':max_speed,'wall_seconds':time.perf_counter()-start})
        print(json.dumps(rows[-1]),flush=True)
out={'results':rows,'limitations':'Two assumed friction values, two time steps and one small servo command; not a real friction measurement, numerical convergence proof or hardware validation.'}
(p/'sector_timestep_friction_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
