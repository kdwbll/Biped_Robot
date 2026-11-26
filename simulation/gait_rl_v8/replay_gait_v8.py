"""第 8 轮双足步态 3D 回放与无头严格真净空验收脚本（不启动训练，不向仓库写入临时压缩包）。

用法（在仓库根目录或本目录均可）：
  python simulation/gait_rl_v8/replay_gait_v8.py               # 打开 3D MuJoCo 窗口回放（镜头自动跟随）
  python simulation/gait_rl_v8/replay_gait_v8.py --headless    # 无头跑 30 秒输出严格真净空验收成绩单
  python simulation/gait_rl_v8/replay_gait_v8.py --headless --seconds 8
"""
import argparse
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

THIS_DIR = Path(__file__).resolve().parent
SIM_DIR = THIS_DIR.parent
DEFAULT_CKPT_DIR = SIM_DIR / "模型检查点" / "biped_walk_round8_20261126"
EXPECTED_FILES = (
    "data",
    "pytorch_variables.pth",
    "policy.pth",
    "policy.optimizer.pth",
    "_stable_baselines3_version",
    "system_info.txt",
)

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from biped_gait_env import BipedGaitV8Env  # noqa: E402


def load_policy_from_unpacked_dir(ckpt_dir: Path) -> PPO:
    """将仓库中展开保存的 SB3 检查点在系统临时目录中组装为只读 ZIP 加载，避免在仓库内残留压缩包。"""
    missing = [name for name in EXPECTED_FILES if not (ckpt_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"检查点目录 {ckpt_dir} 缺少文件：{missing}")
    with tempfile.TemporaryDirectory(prefix="biped_v8_ckpt_") as tmpdir:
        tmp_zip = Path(tmpdir) / "policy.zip"
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in EXPECTED_FILES:
                zf.write(ckpt_dir / name, arcname=name)
        return PPO.load(str(tmp_zip), device="cpu")


def main():
    parser = argparse.ArgumentParser(description="回放或无头验收第 8 轮双足步态策略（不启动训练）。")
    parser.add_argument("--headless", action="store_true", help="无头模式，运行指定时长并打印严格真净空验收成绩单")
    parser.add_argument("--seconds", type=float, default=30.0, help="单局仿真时长（秒，默认 30.0）")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR, help="已展开的 SB3 检查点目录")
    args = parser.parse_args()

    env = BipedGaitV8Env(episode_seconds=args.seconds)
    policy = load_policy_from_unpacked_dir(args.checkpoint_dir)
    if policy.observation_space.shape != env.observation_space.shape:
        raise SystemExit(f"观测维度不匹配：模型 {policy.observation_space.shape} vs 环境 {env.observation_space.shape}")
    print(f"已加载展开检查点：{args.checkpoint_dir}")

    obs, _ = env.reset(seed=0)

    if args.headless:
        total_reward = 0.0
        tilts, yaws, splits = [], [], []
        peak_clear = np.zeros(2)
        stance_dx = np.zeros(2)
        swing_dx = np.zeros(2)
        prev_x = env.data.geom_xpos[env.sole_geom_ids, 0].copy()
        steps = 0
        fell = False
        info = {}

        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            tilts.append(info["tilt_deg"])
            yaws.append(info["yaw_deg"])
            peak_clear = np.maximum(peak_clear, np.array(info["foot_lift_mm"]))
            cur_x = env.data.geom_xpos[env.sole_geom_ids, 0].copy()
            splits.append(abs(float(cur_x[0] - cur_x[1])) * 1000.0)
            dx = (cur_x - prev_x) * 1000.0
            prev_x = cur_x
            for leg in (0, 1):
                if info["foot_force_n"][leg] > 5.0:
                    stance_dx[leg] += dx[leg]
                else:
                    swing_dx[leg] += dx[leg]
            steps += 1
            if steps % 150 == 0 or terminated or truncated:
                fwd = info["net_forward_mm"]
                spd = fwd / max(env.data.time, 1e-6)
                print(
                    f"t={env.data.time:5.2f}s | 前进={fwd:+7.1f}mm ({spd:4.1f}mm/s) | "
                    f"偏航={info['yaw_deg']:+5.1f}° | 倾角={info['tilt_deg']:4.1f}° | "
                    f"真净空={info['foot_lift_mm']}mm | 有效抬脚={info['valid_lift_events']} | "
                    f"交替={info['alt_steps_ok']} 同脚={info['same_leg_lifts']}"
                )
            if terminated or truncated:
                fell = terminated
                break

        tilts = np.array(tilts)
        yaws = np.array(yaws)
        slide_ratio = float(stance_dx.sum()) / max(abs(float(swing_dx.sum())) + abs(float(stance_dx.sum())), 1e-6)
        print("=" * 86)
        print(f"存活 {steps * env.control_dt:.2f}s | 摔倒/越界终止: {fell} | 累计奖励: {total_reward:+.3f}")
        print(
            f"净前进 {info['net_forward_mm']:+.1f} mm | 平均速度 {info['net_forward_mm'] / max(env.data.time, 1e-6):.1f} mm/s | "
            f"最终偏航 {info['yaw_deg']:+.2f}° (最大 |偏航| {np.max(np.abs(yaws)):.2f}°)"
        )
        print(
            f"倾角: 平均 {tilts.mean():.2f}° | 95%分位 {np.percentile(tilts, 95):.2f}° | 最大 {tilts.max():.2f}° | "
            f"最大前后劈叉 {max(splits):.1f} mm"
        )
        print(
            f"真碰撞网格净空峰值: {peak_clear.round(2).tolist()} mm | "
            f"严格有效抬脚(>=2mm,>=60ms): {info['valid_lift_events']} 次 | "
            f"左右交替步: {info['alt_steps_ok']} / 同脚连抬: {info['same_leg_lifts']}"
        )
        print(
            f"拖脚诊断: 腾空前移 {swing_dx.round(1).tolist()} mm | 承重前移(滑动) {stance_dx.round(1).tolist()} mm | "
            f"滑动占比 {slide_ratio * 100:.1f}%"
        )
        return

    print("正在打开 3D 窗口回放第 8 轮步态策略（镜头自动跟随，关闭窗口结束）...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.10, -0.04, 0.12]
        viewer.cam.distance = 0.90
        viewer.cam.azimuth = 125
        viewer.cam.elevation = -15

        episode = 1
        step_i = 0
        total_reward = 0.0
        while viewer.is_running():
            t0 = time.perf_counter()
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_i += 1
            total_reward += reward
            viewer.cam.lookat[0] = float(env.data.xpos[env.torso_id, 0])
            viewer.cam.lookat[1] = float(env.data.xpos[env.torso_id, 1])
            viewer.sync()

            if step_i % 25 == 0 or terminated or truncated:
                print(
                    f"[第{episode}局] t={env.data.time:5.2f}s 前进={info['net_forward_mm']:+7.1f}mm "
                    f"偏航={info['yaw_deg']:+5.1f}° 倾角={info['tilt_deg']:4.1f}° "
                    f"真净空={info['foot_lift_mm']}mm 有效抬脚={info['valid_lift_events']} "
                    f"交替={info['alt_steps_ok']} 总分={total_reward:+.3f}"
                )

            if terminated or truncated:
                episode += 1
                step_i = 0
                total_reward = 0.0
                obs, _ = env.reset(seed=0)

            left = env.control_dt - (time.perf_counter() - t0)
            if left > 0:
                time.sleep(left)


if __name__ == "__main__":
    main()
