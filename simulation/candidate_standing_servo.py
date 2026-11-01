"""Simulation-only PI-D servo experiment. Parameters are not hardware settings."""
import numpy as np
import mujoco
from theoretical_servo import TheoreticalServoController

class CandidateStandingServo(TheoreticalServoController):
    def __init__(self, model, data, spec, ki=1.5, integral_limit_nm=1.0,
                 kp_by_servo=None, ki_by_servo=None, hip_damping=0.01):
        super().__init__(model, data, spec)
        self.ki = float(ki)
        self.kp_by_servo = np.full(5, spec.kp_nm_per_rad) if kp_by_servo is None else np.asarray(kp_by_servo,dtype=float)
        self.ki_by_servo = np.full(5, self.ki) if ki_by_servo is None else np.asarray(ki_by_servo,dtype=float)
        if self.kp_by_servo.shape != (5,) or self.ki_by_servo.shape != (5,) or not np.isfinite(self.kp_by_servo).all() or not np.isfinite(self.ki_by_servo).all():
            raise ValueError('Expected five finite simulated gains')
        self.integral_limit_nm = float(integral_limit_nm)
        self.hip_damping = float(hip_damping)
        if not np.isfinite(self.hip_damping) or self.hip_damping < 0:
            raise ValueError('hip_damping must be finite and nonnegative')
        self.integral_torque = np.zeros(5)

    def step(self):
        s = self.spec
        now = float(self.data.time)
        if now + 1e-12 >= self.next_update:
            fraction = (self.pending_us-s.center_us)/(s.max_us-s.min_us)
            command = self.initial_positions + fraction*s.travel_rad
            step_limit = s.no_load_speed_rad_s*s.update_interval_s
            self.target_positions += np.clip(command-self.target_positions, -step_limit, step_limit)
            self.next_update = now+s.update_interval_s
        error = self.target_positions-self.data.actuator_length
        velocity = self.data.actuator_velocity.copy()
        damping = np.array([s.kd_nm_per_rad_s]*4+[self.hip_damping])
        raw = self.kp_by_servo*error-damping*velocity+self.integral_torque
        available = np.where(raw*velocity > 0,
                             s.stall_torque_nm*np.maximum(0.,1.-np.abs(velocity)/s.no_load_speed_rad_s),
                             s.stall_torque_nm)
        torque = np.clip(raw, -available, available)
        # Stop accumulating when it would push farther into the instantaneous torque limit.
        permit = (np.abs(raw) <= available) | ((raw-torque)*error < 0)
        self.integral_torque = np.clip(self.integral_torque + permit*self.ki_by_servo*error*self.model.opt.timestep,
                                       -self.integral_limit_nm, self.integral_limit_nm)
        self.data.ctrl[:] = torque
        mujoco.mj_step(self.model,self.data)
