"""Isolated numerical experiment: fifth servo damping differs from legacy model."""
import numpy as np,mujoco
from theoretical_servo import TheoreticalServoController

class StabilizedServo(TheoreticalServoController):
 def step(self):
  s=self.spec;now=float(self.data.time)
  if now+1e-12>=self.next_update:
   fraction=(self.pending_us-s.center_us)/(s.max_us-s.min_us)
   commanded=self.initial_positions+fraction*s.travel_rad
   limit=s.no_load_speed_rad_s*s.update_interval_s
   self.target_positions+=np.clip(commanded-self.target_positions,-limit,limit)
   self.next_update=now+s.update_interval_s
  vel=self.data.actuator_velocity.copy();kd=np.full(5,s.kd_nm_per_rad_s);kd[4]=.01
  raw=s.kp_nm_per_rad*(self.target_positions-self.data.actuator_length)-kd*vel
  available=np.where(raw*vel>0,s.stall_torque_nm*np.maximum(0.,1.-np.abs(vel)/s.no_load_speed_rad_s),s.stall_torque_nm)
  self.data.ctrl[:]=np.clip(raw,-available,available);mujoco.mj_step(self.model,self.data)
