"""第 8 轮双足步态强化学习环境（CPU 单环境版，19 维观测 / 5 维动作，真碰撞网格净空 + 无硬时钟交替真抬脚）。"""
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import robot_loader as robot  # noqa: E402

SOLE_GEOMS = ("L0_foot_proxy", "L1_foot_proxy")


class BipedGaitV8Env(gym.Env):
    """3D 五舵机完整闭链双足步态环境（仅限仿真研究，未经实物硬件标定）。"""

    def __init__(self, episode_seconds=8.0):
        super().__init__()
        self.model, self.data = robot.load(hang=False)
        self.torso_id = self.model.body(robot.TORSO_BODY).id
        self.foot_ids = [self.model.body(name).id for name in robot.FOOT_BODIES]
        self.sole_geom_ids = [self.model.geom(name).id for name in SOLE_GEOMS]
        self.foot_mesh_gids, self.foot_hull_verts = robot.foot_mesh_hull_verts(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        # 19 维观测：重力方向(3) + 机身角速度(3) + 偏航角yaw_rad(1) + 上一拍动作(5) + 舵机目标角偏移(5) + 节律sin/cos(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

        self.control_dt = 0.02
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        self.max_steps = round(episode_seconds / self.control_dt)
        self.amp_us = np.array([155.0, 155.0, 155.0, 155.0, 25.0], dtype=np.float32)
        self.slew_us = 5.0
        self.gait_period_s = 1.4
        self.lift_cooldown_steps = 15

        self.servo = None
        self.pwm_us = np.full(5, robot.CENTER_US)
        self.prev_action = np.zeros(5, dtype=np.float32)
        self.initial_x = 0.0
        self.initial_z = 0.0
        self.initial_sole_x = np.zeros(2)
        self.consec_lift = np.zeros(2, dtype=np.int32)
        self.in_valid_lift = np.zeros(2, dtype=bool)
        self.last_lift_step = np.full(2, -100, dtype=np.int32)
        self.last_event_sole_x = np.zeros(2)
        self.last_lifted_leg = -1
        self.valid_lift_events = np.zeros(2, dtype=np.int32)
        self.alt_steps_ok = 0
        self.same_leg_lifts = 0
        self.step_count = 0
        self.phase = 0.0

    def _get_obs(self):
        rot = self.data.xmat[self.torso_id].reshape(3, 3)
        gravity_in_body = rot.T @ np.array([0.0, 0.0, -1.0])
        vel6 = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.torso_id, vel6, 1)
        ang_vel = vel6[:3]
        yaw_rad = np.array([np.arctan2(rot[1, 0], rot[0, 0])], dtype=np.float32)
        servo_targets = self.servo.target_positions - self.servo.initial_positions
        phase_angle = 2.0 * np.pi * self.phase
        phase_obs = np.array([np.sin(phase_angle), np.cos(phase_angle)], dtype=np.float32)
        return np.concatenate(
            [gravity_in_body, ang_vel, yaw_rad, self.prev_action, servo_targets, phase_obs]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.servo = robot.make_servo(self.model, self.data)
        self.pwm_us = np.full(5, robot.CENTER_US)
        self.prev_action = np.zeros(5, dtype=np.float32)
        self.initial_x = float(self.data.xpos[self.torso_id, 0])
        self.initial_z = float(self.data.xpos[self.torso_id, 2])
        self.initial_sole_x = self.data.geom_xpos[self.sole_geom_ids, 0].copy()
        self.consec_lift = np.zeros(2, dtype=np.int32)
        self.in_valid_lift = np.zeros(2, dtype=bool)
        self.last_lift_step = np.full(2, -100, dtype=np.int32)
        self.last_event_sole_x = self.initial_sole_x.copy()
        self.last_lifted_leg = -1
        self.valid_lift_events = np.zeros(2, dtype=np.int32)
        self.alt_steps_ok = 0
        self.same_leg_lifts = 0
        self.step_count = 0
        self.phase = 0.0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        x_before = float(self.data.xpos[self.torso_id, 0])
        y_before = float(self.data.xpos[self.torso_id, 1])
        soles_before = self.data.geom_xpos[self.sole_geom_ids].copy()

        desired_pwm = robot.CENTER_US + self.amp_us * action
        self.pwm_us = self.pwm_us + np.clip(desired_pwm - self.pwm_us, -self.slew_us, self.slew_us)
        self.servo.set_pulse_us(self.pwm_us)

        for _ in range(self.substeps):
            self.servo.step()
        mujoco.mj_forward(self.model, self.data)

        self.step_count += 1
        self.phase = (self.step_count * self.control_dt / self.gait_period_s) % 1.0

        action_jerk = float(np.mean(np.square(action - self.prev_action)))
        self.prev_action = action.copy()
        obs = self._get_obs()

        x_after = float(self.data.xpos[self.torso_id, 0])
        forward_vel = (x_after - x_before) / self.control_dt
        side_vel = abs(float(self.data.xpos[self.torso_id, 1]) - y_before) / self.control_dt
        net_forward_m = x_after - self.initial_x
        tilt_deg = robot.torso_tilt_deg(self.data)
        rot = self.data.xmat[self.torso_id].reshape(3, 3)
        yaw_deg = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
        yaw_rate = abs(float(obs[5]))
        roll_pitch_speed = float(np.hypot(obs[3], obs[4]))
        height_drop = max(0.0, self.initial_z - float(self.data.xpos[self.torso_id, 2]))

        soles_after = self.data.geom_xpos[self.sole_geom_ids]
        true_clearance = robot.foot_true_clearance_m(self.data, self.foot_mesh_gids, self.foot_hull_verts)
        sole_vel_x = (soles_after[:, 0] - soles_before[:, 0]) / self.control_dt
        sole_forward_mm = (soles_after[:, 0] - self.initial_sole_x) * 1000.0
        foot_force = np.array(robot.foot_forces(self.model, self.data), dtype=np.float64)

        upright_w = float(np.exp(-((max(0.0, tilt_deg - 22.0) / 7.0) ** 2)))
        straight_w = float(np.exp(-((yaw_deg / 12.0) ** 2)))
        height_w = float(np.exp(-((height_drop / 0.025) ** 2)))
        alive_term = 0.20 * upright_w * straight_w * height_w

        stance_w = np.clip((foot_force - 4.0) / 8.0, 0.0, 1.0)
        excess_drag_x = float(np.sum(stance_w * np.maximum(0.0, np.abs(sole_vel_x) - 0.022)))
        clean_drag_gate = float(0.35 + 0.65 * np.exp(-excess_drag_x / 0.035))

        v_norm = float(np.clip(forward_vel / 0.040, -1.0, 1.0))
        speed_cap = float(np.clip(1.0 - max(0.0, forward_vel - 0.055) / 0.065, 0.25, 1.0))
        if v_norm >= 0.0:
            vel_term = 1.6 * v_norm * speed_cap * clean_drag_gate * straight_w
        else:
            vel_term = 1.6 * v_norm * clean_drag_gate

        gait_term = 0.0
        step_bonus = 0.0
        for leg in (0, 1):
            other = 1 - leg
            sup = float(np.clip(foot_force[other] / 10.0, 0.0, 1.0))
            unl = float(np.clip(1.0 - foot_force[leg] / 3.0, 0.0, 1.0))
            lift_h = float(np.clip((true_clearance[leg] - 0.0005) / 0.0055, 0.0, 1.0))
            fwd_swing = float(np.clip(sole_vel_x[leg] / 0.06, 0.0, 1.0))
            is_next_alt = (self.last_lifted_leg == -1) or (leg != self.last_lifted_leg)
            alt_mult = 1.0 if is_next_alt else 0.25
            gait_term += sup * unl * lift_h * (4.0 + 1.5 * fwd_swing) * alt_mult * straight_w

            valid_frame = (
                (true_clearance[leg] >= 0.002)
                and (foot_force[leg] < 0.5)
                and (foot_force[other] > 5.0)
                and (tilt_deg < 26.0)
            )
            if valid_frame:
                self.consec_lift[leg] += 1
                if (
                    self.consec_lift[leg] >= 3
                    and (not self.in_valid_lift[leg])
                    and (self.step_count - self.last_lift_step[leg] >= self.lift_cooldown_steps)
                ):
                    self.in_valid_lift[leg] = True
                    self.last_lift_step[leg] = self.step_count
                    self.valid_lift_events[leg] += 1
                    sole_dx = float(soles_after[leg, 0] - self.last_event_sole_x[leg])
                    self.last_event_sole_x[leg] = float(soles_after[leg, 0])
                    if is_next_alt:
                        step_bonus += 1.20 + (0.60 if sole_dx >= 0.010 else 0.0)
                        self.alt_steps_ok += 1
                    else:
                        step_bonus += 0.15
                        self.same_leg_lifts += 1
                    self.last_lifted_leg = leg
            else:
                self.consec_lift[leg] = 0
                self.in_valid_lift[leg] = False

        foot_split_excess = max(0.0, abs(float(soles_after[0, 0] - soles_after[1, 0])) - 0.045)
        excess_tilt = float(np.clip((tilt_deg - 23.0) / 5.0, 0.0, 1.0))
        reg_penalty = (
            0.08 * action_jerk
            + 0.12 * min(yaw_rate, 1.5)
            + 0.04 * min(roll_pitch_speed, 2.5)
            + 0.15 * min(side_vel, 0.5)
            + 2.5 * min(foot_split_excess, 0.10)
            + 0.30 * excess_tilt
        )

        step_rate = max(-2.5, alive_term + vel_term + gait_term - reg_penalty)
        reward = float(self.control_dt * step_rate + step_bonus)

        terminated = bool(tilt_deg > 28.0 or height_drop > 0.05 or abs(yaw_deg) > 35.0)
        if terminated:
            reward -= 2.0
        truncated = bool(self.step_count >= self.max_steps and not terminated)

        info = {
            "forward_vel_mps": forward_vel,
            "net_forward_mm": net_forward_m * 1000.0,
            "sole_forward_mm": sole_forward_mm.round(1).tolist(),
            "tilt_deg": tilt_deg,
            "yaw_deg": yaw_deg,
            "height_drop_mm": height_drop * 1000.0,
            "foot_lift_mm": (true_clearance * 1000.0).round(2).tolist(),
            "foot_force_n": foot_force.round(1).tolist(),
            "drag_x_speed": round(excess_drag_x, 4),
            "clean_drag_gate": round(clean_drag_gate, 3),
            "valid_lift_events": self.valid_lift_events.tolist(),
            "alt_steps_ok": self.alt_steps_ok,
            "same_leg_lifts": self.same_leg_lifts,
            "last_lifted_leg": self.last_lifted_leg,
            "pwm_us": self.pwm_us.round(1).tolist(),
        }
        return obs, reward, terminated, truncated, info
