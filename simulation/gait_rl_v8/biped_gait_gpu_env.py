"""第 8 轮双足步态 MuJoCo Warp GPU 并行环境（与 biped_gait_env.py 逻辑严格对齐）。"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import robot_loader as robot  # noqa: E402

SOLE_GEOMS = ("L0_foot_proxy", "L1_foot_proxy")


def _init_warp_ascii_safe():
    """在中文路径环境下把 Warp native 头文件镜像到 %TEMP% 下的纯 ASCII 路径再编译 CUDA 内核。"""
    import warp as wp
    import warp._src.build as wp_build

    ascii_home = os.path.join(tempfile.gettempdir(), "biped_warp_home")
    dst_native = os.path.join(ascii_home, "native")
    src_native = os.path.join(wp_build.warp_home, "native")
    if not os.path.exists(os.path.join(dst_native, "builtin.h")):
        shutil.copytree(src_native, dst_native, dirs_exist_ok=True)
    wp_build.warp_home = ascii_home
    wp.config.kernel_cache_dir = os.path.join(tempfile.gettempdir(), "biped_warp_cache")
    wp.init()
    return wp


class BipedGaitV8GpuVecEnv(VecEnv):
    """基于 MuJoCo Warp 的 256 并行 3D 双足步态训练环境（仅限仿真，非实物安全控制）。"""

    def __init__(self, n_envs=256, episode_seconds=8.0):
        self.wp = _init_warp_ascii_safe()
        import mujoco_warp as mjw
        from sector_warp_servo_kernels import servo_step, update_pwm_targets
        from sector_warp_guard import restore_masked_state

        self.mjw = mjw
        self._servo_step = servo_step
        self._update_pwm_targets = update_pwm_targets
        self._restore_masked_state = restore_masked_state

        self.n_envs = int(n_envs)
        self.render_mode = None
        self.control_dt = 0.02
        self.max_steps = round(episode_seconds / self.control_dt)
        self.gait_period_s = 1.4
        self.lift_cooldown_steps = 15
        self.amp_us = np.array([155.0, 155.0, 155.0, 155.0, 25.0], dtype=np.float32)
        self.slew_us = 5.0

        self.model, self.cpu_data = robot.load(hang=False)
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        self.torso_id = self.model.body(robot.TORSO_BODY).id
        self.foot_ids = [self.model.body(name).id for name in robot.FOOT_BODIES]
        self.sole_geom_ids = [self.model.geom(name).id for name in SOLE_GEOMS]
        self.foot_mesh_gids, self.foot_hull_verts = robot.foot_mesh_hull_verts(self.model)
        self.foot_hull_verts_T = [np.ascontiguousarray(v.T, dtype=np.float32) for v in self.foot_hull_verts]
        self.geom_bodyid = np.asarray(self.model.geom_bodyid, dtype=np.int32)
        cpu_servo = robot.make_servo(self.model, self.cpu_data)

        self.initial_q = self.cpu_data.qpos.astype(np.float32).copy()
        self.initial_target = cpu_servo.initial_positions.astype(np.float32).copy()
        self.initial_x = float(self.cpu_data.xpos[self.torso_id, 0])
        self.initial_z = float(self.cpu_data.xpos[self.torso_id, 2])
        self.initial_sole_x = self.cpu_data.geom_xpos[self.sole_geom_ids, 0].astype(np.float32).copy()

        wp = self.wp
        with wp.ScopedDevice("cuda:0"):
            self.gm = mjw.put_model(self.model)
            self.gd = mjw.put_data(
                self.model,
                self.cpu_data,
                nworld=self.n_envs,
                nconmax=64,
                njmax=512,
            )
            mjw.forward(self.gm, self.gd)

            init_tile = np.tile(self.initial_target, (self.n_envs, 1))
            self.initial_q_dev = wp.array(self.initial_q, dtype=wp.float32, device="cuda:0")
            self.initial_target_dev = wp.array(self.initial_target, dtype=wp.float32, device="cuda:0")
            self.init_targets_dev = wp.array(init_tile, dtype=wp.float32, device="cuda:0")
            self.targets_dev = wp.array(init_tile, dtype=wp.float32, device="cuda:0")
            self.integral_dev = wp.zeros((self.n_envs, 5), dtype=wp.float32, device="cuda:0")
            self.flags_dev = wp.zeros(self.n_envs, dtype=wp.int32, device="cuda:0")

            with wp.ScopedCapture() as cap:
                wp.launch(
                    self._servo_step,
                    dim=(self.n_envs, 5),
                    inputs=[
                        self.targets_dev,
                        self.integral_dev,
                        self.gd.actuator_length,
                        self.gd.actuator_velocity,
                        self.gd.ctrl,
                        self.model.opt.timestep,
                        0.01,
                    ],
                )
                mjw.step(self.gm, self.gd)
            self.step_graph = cap.graph
            wp.synchronize()

        self.pwm_us = np.full((self.n_envs, 5), robot.CENTER_US, dtype=np.float32)
        self.prev_action = np.zeros((self.n_envs, 5), dtype=np.float32)
        self.step_counts = np.zeros(self.n_envs, dtype=np.int32)
        self.phases = np.zeros(self.n_envs, dtype=np.float32)
        self.consec_lift = np.zeros((self.n_envs, 2), dtype=np.int32)
        self.in_valid_lift = np.zeros((self.n_envs, 2), dtype=bool)
        self.last_lift_step = np.full((self.n_envs, 2), -100, dtype=np.int32)
        self.last_event_sole_x = np.tile(self.initial_sole_x, (self.n_envs, 1)).astype(np.float32)
        self.last_lifted_leg = np.full(self.n_envs, -1, dtype=np.int32)
        self._pending_actions = None

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        super().__init__(self.n_envs, obs_space, act_space)

    def _get_obs(self):
        rot = self.gd.xmat.numpy()[:, self.torso_id]
        irot = self.gd.ximat.numpy()[:, self.torso_id]
        world_omega = self.gd.cvel.numpy()[:, self.torso_id, :3]
        gravity_in_body = -rot[:, 2, :]
        ang_vel = np.einsum("nji,nj->ni", irot, world_omega)
        yaw_rad = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])[:, None]
        servo_targets = self.targets_dev.numpy() - self.initial_target[None, :]
        phase_angle = 2.0 * np.pi * self.phases
        phase_obs = np.stack([np.sin(phase_angle), np.cos(phase_angle)], axis=1)
        return np.concatenate(
            [gravity_in_body, ang_vel, yaw_rad, self.prev_action, servo_targets, phase_obs], axis=1
        ).astype(np.float32)

    def _true_foot_clearance(self, geom_xpos, geom_xmat):
        out = np.empty((self.n_envs, 2), dtype=np.float32)
        for leg, (gid, verts_T) in enumerate(zip(self.foot_mesh_gids, self.foot_hull_verts_T)):
            pos_z = geom_xpos[:, gid, 2]
            rot_z = geom_xmat[:, gid, 2, :]
            z_min = pos_z + np.min(rot_z @ verts_T, axis=1)
            out[:, leg] = np.maximum(0.0, z_min)
        return out

    def _foot_forces(self):
        forces = np.zeros((self.n_envs, 2), dtype=np.float32)
        n = int(self.gd.nacon.numpy()[0])
        if n <= 0:
            return forces
        wp = self.wp
        with wp.ScopedDevice("cuda:0"):
            ids = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device="cuda:0")
            force_dev = wp.zeros(n, dtype=wp.spatial_vector, device="cuda:0")
            self.mjw.contact_force(self.gm, self.gd, ids, False, force_dev)
            wp.synchronize()

        normal_f = np.maximum(0.0, force_dev.numpy()[:, 0])
        geom = self.gd.contact.geom.numpy()[:n]
        world = self.gd.contact.worldid.numpy()[:n]

        valid = (world >= 0) & (world < self.n_envs) & (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
        if not np.any(valid):
            return forces
        b1 = self.geom_bodyid[geom[valid, 0]]
        b2 = self.geom_bodyid[geom[valid, 1]]
        w = world[valid]
        nf = normal_f[valid]

        for leg, foot_body_id in enumerate(self.foot_ids):
            hit = ((b1 == 0) & (b2 == foot_body_id)) | ((b2 == 0) & (b1 == foot_body_id))
            if np.any(hit):
                np.add.at(forces[:, leg], w[hit], nf[hit])
        return forces

    def _reset_mask(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return
        wp = self.wp
        with wp.ScopedDevice("cuda:0"):
            wm = wp.array(mask, dtype=wp.bool, device="cuda:0")
            self.mjw.reset_data(self.gm, self.gd, wm)
            wp.launch(
                self._restore_masked_state,
                dim=self.n_envs,
                inputs=[
                    self.gd.qpos, self.gd.qvel, self.gd.ctrl,
                    self.targets_dev, self.integral_dev, self.flags_dev,
                    wm, self.initial_q_dev, self.initial_target_dev,
                    self.model.nq, self.model.nv, self.model.nu,
                ],
            )
            self.mjw.forward(self.gm, self.gd)
            wp.synchronize()
        self.pwm_us[mask] = robot.CENTER_US
        self.prev_action[mask] = 0.0
        self.step_counts[mask] = 0
        self.phases[mask] = 0.0
        self.consec_lift[mask] = 0
        self.in_valid_lift[mask] = False
        self.last_lift_step[mask] = -100
        self.last_event_sole_x[mask] = self.initial_sole_x[None, :]
        self.last_lifted_leg[mask] = -1

    def reset(self):
        self._reset_mask(np.ones(self.n_envs, dtype=bool))
        return self._get_obs()

    def step_async(self, actions):
        self._pending_actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)

    def step_wait(self):
        action = self._pending_actions
        self._pending_actions = None

        pos_before = self.gd.xpos.numpy()
        x_before = pos_before[:, self.torso_id, 0].copy()
        y_before = pos_before[:, self.torso_id, 1].copy()
        soles_before = self.gd.geom_xpos.numpy()[:, self.sole_geom_ids].copy()

        desired_pwm = robot.CENTER_US + self.amp_us[None, :] * action
        self.pwm_us = self.pwm_us + np.clip(desired_pwm - self.pwm_us, -self.slew_us, self.slew_us)

        wp = self.wp
        with wp.ScopedDevice("cuda:0"):
            pwm_dev = wp.array(self.pwm_us, dtype=wp.float32, device="cuda:0")
            wp.launch(
                self._update_pwm_targets,
                dim=(self.n_envs, 5),
                inputs=[self.init_targets_dev, self.targets_dev, pwm_dev],
            )
            for _ in range(self.substeps):
                wp.capture_launch(self.step_graph)
            self.mjw.forward(self.gm, self.gd)
            wp.synchronize()

        self.step_counts += 1
        self.phases = ((self.step_counts * self.control_dt) / self.gait_period_s) % 1.0

        action_jerk = np.mean(np.square(action - self.prev_action), axis=1)
        self.prev_action = action.copy()
        obs = self._get_obs()

        torso_pos = self.gd.xpos.numpy()[:, self.torso_id]
        torso_rot = self.gd.xmat.numpy()[:, self.torso_id]
        all_geom_xpos = self.gd.geom_xpos.numpy()
        all_geom_xmat = self.gd.geom_xmat.numpy()
        soles_after = all_geom_xpos[:, self.sole_geom_ids]

        forward_vel = (torso_pos[:, 0] - x_before) / self.control_dt
        side_vel = np.abs(torso_pos[:, 1] - y_before) / self.control_dt
        net_forward_mm = (torso_pos[:, 0] - self.initial_x) * 1000.0
        upright_cos = np.clip(torso_rot[:, 2, 2], -1.0, 1.0)
        tilt_deg = np.degrees(np.arccos(upright_cos))
        yaw_deg = np.degrees(np.arctan2(torso_rot[:, 1, 0], torso_rot[:, 0, 0]))
        yaw_rate = np.abs(obs[:, 5])
        roll_pitch_speed = np.hypot(obs[:, 3], obs[:, 4])
        height_drop = np.maximum(0.0, self.initial_z - torso_pos[:, 2])

        true_clearance = self._true_foot_clearance(all_geom_xpos, all_geom_xmat)
        sole_vel_x = (soles_after[:, :, 0] - soles_before[:, :, 0]) / self.control_dt
        foot_force = self._foot_forces()

        upright_w = np.exp(-((np.maximum(0.0, tilt_deg - 22.0) / 7.0) ** 2))
        straight_w = np.exp(-((yaw_deg / 12.0) ** 2))
        height_w = np.exp(-((height_drop / 0.025) ** 2))
        alive_term = 0.20 * upright_w * straight_w * height_w

        stance_w = np.clip((foot_force - 4.0) / 8.0, 0.0, 1.0)
        excess_drag_x = np.sum(stance_w * np.maximum(0.0, np.abs(sole_vel_x) - 0.022), axis=1)
        clean_drag_gate = 0.35 + 0.65 * np.exp(-excess_drag_x / 0.035)

        v_norm = np.clip(forward_vel / 0.040, -1.0, 1.0)
        speed_cap = np.clip(1.0 - np.maximum(0.0, forward_vel - 0.055) / 0.065, 0.25, 1.0)
        vel_term = np.where(
            v_norm >= 0.0,
            1.6 * v_norm * speed_cap * clean_drag_gate * straight_w,
            1.6 * v_norm * clean_drag_gate,
        )

        gait_term = np.zeros(self.n_envs, dtype=np.float32)
        step_bonus = np.zeros(self.n_envs, dtype=np.float32)
        for leg in (0, 1):
            other = 1 - leg
            sup = np.clip(foot_force[:, other] / 10.0, 0.0, 1.0)
            unl = np.clip(1.0 - foot_force[:, leg] / 3.0, 0.0, 1.0)
            lift_h = np.clip((true_clearance[:, leg] - 0.0005) / 0.0055, 0.0, 1.0)
            fwd_swing = np.clip(sole_vel_x[:, leg] / 0.06, 0.0, 1.0)
            is_next_alt = (self.last_lifted_leg == -1) | (self.last_lifted_leg != leg)
            alt_mult = np.where(is_next_alt, 1.0, 0.25)
            gait_term += (sup * unl * lift_h * (4.0 + 1.5 * fwd_swing) * alt_mult * straight_w).astype(np.float32)

            valid_frame = (
                (true_clearance[:, leg] >= 0.002)
                & (foot_force[:, leg] < 0.5)
                & (foot_force[:, other] > 5.0)
                & (tilt_deg < 26.0)
            )
            self.consec_lift[:, leg] = np.where(valid_frame, self.consec_lift[:, leg] + 1, 0)
            self.in_valid_lift[:, leg] = np.where(valid_frame, self.in_valid_lift[:, leg], False)

            trigger = (
                valid_frame
                & (self.consec_lift[:, leg] >= 3)
                & (~self.in_valid_lift[:, leg])
                & ((self.step_counts - self.last_lift_step[:, leg]) >= self.lift_cooldown_steps)
            )
            if np.any(trigger):
                self.in_valid_lift[trigger, leg] = True
                self.last_lift_step[trigger, leg] = self.step_counts[trigger]
                sole_dx = soles_after[:, leg, 0] - self.last_event_sole_x[:, leg]
                self.last_event_sole_x[trigger, leg] = soles_after[trigger, leg, 0]
                alt_bonus = 1.20 + np.where(sole_dx >= 0.010, 0.60, 0.0)
                bonus_val = np.where(is_next_alt, alt_bonus, 0.15)
                step_bonus += np.where(trigger, bonus_val, 0.0).astype(np.float32)
                self.last_lifted_leg[trigger] = leg

        foot_split_excess = np.maximum(0.0, np.abs(soles_after[:, 0, 0] - soles_after[:, 1, 0]) - 0.045)
        excess_tilt = np.clip((tilt_deg - 23.0) / 5.0, 0.0, 1.0)
        reg_penalty = (
            0.08 * action_jerk
            + 0.12 * np.minimum(yaw_rate, 1.5)
            + 0.04 * np.minimum(roll_pitch_speed, 2.5)
            + 0.15 * np.minimum(side_vel, 0.5)
            + 2.5 * np.minimum(foot_split_excess, 0.10)
            + 0.30 * excess_tilt
        )

        step_rate = np.maximum(-2.5, alive_term + vel_term + gait_term - reg_penalty)
        rewards = (self.control_dt * step_rate + step_bonus).astype(np.float32)

        nonfinite = ~np.isfinite(torso_pos).all(axis=1)
        terminated = nonfinite | (tilt_deg > 28.0) | (height_drop > 0.05) | (np.abs(yaw_deg) > 35.0)
        rewards[terminated] -= 2.0
        truncated = (self.step_counts >= self.max_steps) & ~terminated
        dones = terminated | truncated

        infos = [
            {
                "forward_vel_mps": float(forward_vel[i]),
                "net_forward_mm": float(net_forward_mm[i]),
                "tilt_deg": float(tilt_deg[i]),
                "TimeLimit.truncated": bool(truncated[i]),
            }
            for i in range(self.n_envs)
        ]

        if np.any(dones):
            for i in np.flatnonzero(dones):
                infos[i]["terminal_observation"] = obs[i].copy()
            self._reset_mask(dones)
            obs = self._get_obs()

        return obs, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
