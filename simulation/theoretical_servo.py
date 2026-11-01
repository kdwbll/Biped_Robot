"""Assumed 6 V hobby-servo model for simulation only; not a hardware driver."""
from dataclasses import dataclass
import numpy as np
import mujoco

@dataclass(frozen=True)
class ServoSpec:
    min_us: float = 500.0
    center_us: float = 1500.0
    max_us: float = 2500.0
    travel_rad: float = np.deg2rad(270.0)
    no_load_speed_rad_s: float = np.deg2rad(60.0)/0.26
    stall_torque_nm: float = 24.0 * 0.0980665
    update_interval_s: float = 0.02
    kp_nm_per_rad: float = 1.0  # unmeasured simulated controller
    kd_nm_per_rad_s: float = 0.08  # unmeasured simulated controller

class TheoreticalServoController:
    def __init__(self, model, data, spec=ServoSpec()):
        if model.nu != 5: raise ValueError('This controller expects five candidate actuators')
        self.model, self.data, self.spec = model, data, spec
        mujoco.mj_forward(model, data)
        self.initial_positions = data.actuator_length.copy()
        self.target_positions = self.initial_positions.copy()
        self.pending_us = np.full(5, spec.center_us)
        self.next_update = float(data.time)
        # These model limits are on simulated servo torque (N m), not PWM.
        model.actuator_ctrllimited[:] = 1
        model.actuator_ctrlrange[:,0] = -spec.stall_torque_nm
        model.actuator_ctrlrange[:,1] = spec.stall_torque_nm

    def set_pulse_us(self, values):
        v = np.asarray(values, dtype=float)
        if v.shape != (5,) or not np.isfinite(v).all(): raise ValueError('Expected five finite pulse widths')
        self.pending_us = np.clip(v, self.spec.min_us, self.spec.max_us)

    def step(self):
        s=self.spec; now=float(self.data.time)
        if now + 1e-12 >= self.next_update:
            fraction=(self.pending_us-s.center_us)/(s.max_us-s.min_us)
            commanded=self.initial_positions + fraction*s.travel_rad
            limit=s.no_load_speed_rad_s*s.update_interval_s
            self.target_positions += np.clip(commanded-self.target_positions,-limit,limit)
            self.next_update=now+s.update_interval_s
        vel=self.data.actuator_velocity.copy()
        raw=s.kp_nm_per_rad*(self.target_positions-self.data.actuator_length)-s.kd_nm_per_rad_s*vel
        driving=raw*vel>0
        available=np.where(driving,s.stall_torque_nm*np.maximum(0.0,1.0-np.abs(vel)/s.no_load_speed_rad_s),s.stall_torque_nm)
        self.data.ctrl[:]=np.clip(raw,-available,available)
        mujoco.mj_step(self.model,self.data)
