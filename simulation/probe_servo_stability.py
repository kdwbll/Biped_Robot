"""Explore numerics of the fifth candidate actuator; not calibrated hardware."""
import json
from dataclasses import replace
import numpy as np,mujoco
from biped_env import BipedEnv
from theoretical_servo import TheoreticalServoController,ServoSpec
rows=[]
for kd5 in (.08,.01,.003,.001,0.):
 e=BipedEnv(episode_seconds=2.);e.reset(seed=0);s=ServoSpec();controller=e.controller
 peak=0.;limit_count=0; fifth=[]
 for frame in range(50):
  phase=2*np.pi*frame*.02
  actions=np.array([-.5*max(0,np.sin(phase)),.5*max(0,np.sin(phase)),.5*max(0,-np.sin(phase)),-.5*max(0,-np.sin(phase)),.5*np.sin(phase)])
  controller.set_pulse_us(1500+180*actions)
  for _ in range(e.steps_per_control):
   now=float(e.data.time)
   if now+1e-12>=controller.next_update:
    fraction=(controller.pending_us-s.center_us)/(s.max_us-s.min_us);cmd=controller.initial_positions+fraction*s.travel_rad
    controller.target_positions+=np.clip(cmd-controller.target_positions,-s.no_load_speed_rad_s*s.update_interval_s,s.no_load_speed_rad_s*s.update_interval_s)
    controller.next_update=now+s.update_interval_s
   vel=e.data.actuator_velocity.copy();kd=np.full(5,s.kd_nm_per_rad_s);kd[4]=kd5
   raw=s.kp_nm_per_rad*(controller.target_positions-e.data.actuator_length)-kd*vel
   available=np.where(raw*vel>0,s.stall_torque_nm*np.maximum(0,1-np.abs(vel)/s.no_load_speed_rad_s),s.stall_torque_nm)
   e.data.ctrl[:]=np.clip(raw,-available,available);mujoco.mj_step(e.model,e.data)
   peak=max(peak,float(abs(e.data.qvel[-1])));limit_count+=int(abs(e.data.ctrl[-1])>s.stall_torque_nm*.99)
  fifth.append(float(e.data.qvel[-1]))
 row=dict(kd5=kd5,peak_shaft_speed_rad_s=peak,mean_abs_shaft_speed_rad_s=float(np.mean(np.abs(fifth))),
          saturated_substeps=limit_count,finite=bool(np.isfinite(e.data.qpos).all() and np.isfinite(e.data.qvel).all()))
 rows.append(row);e.close();print(json.dumps(row),flush=True)
with open('servo_stability_probe.json','w',encoding='utf8') as f:json.dump(rows,f,indent=2)
