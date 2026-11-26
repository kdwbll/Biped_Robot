"""第 8 轮 MuJoCo Warp GPU 并行步态训练入口（仅在用户明确要求时手动运行，默认输出到系统临时目录，不污染仓库）。

示例（仅当需要复现或续训时执行）：
  python simulation/gait_rl_v8/train_gait_v8_gpu.py --output-dir "%TEMP%/biped_r8_run"
  python simulation/gait_rl_v8/train_gait_v8_gpu.py --resume-dir simulation/模型检查点/biped_walk_round8_20261126 --episode-seconds 12 --output-dir "%TEMP%/biped_r9_run"
"""
import argparse
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

THIS_DIR = Path(__file__).resolve().parent
SIM_DIR = THIS_DIR.parent
DEFAULT_WARMSTART_DIR = SIM_DIR / "模型检查点" / "biped_walk_round4_20261125"
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

from biped_gait_gpu_env import BipedGaitV8GpuVecEnv  # noqa: E402


def _load_unpacked(ckpt_dir: Path, custom_objects=None) -> PPO:
    with tempfile.TemporaryDirectory(prefix="biped_load_") as tmpdir:
        tmp_zip = Path(tmpdir) / "ckpt.zip"
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in EXPECTED_FILES:
                zf.write(ckpt_dir / name, arcname=name)
        return PPO.load(str(tmp_zip), device="cpu", custom_objects=custom_objects)


def save_unpacked(policy: PPO, dst_dir: Path):
    """按仓库规范解包保存检查点（不留 .zip）。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="biped_save_") as tmpdir:
        tmp_zip = Path(tmpdir) / "model.zip"
        policy.save(str(tmp_zip))
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(dst_dir)


def warmstart_actor_18_to_19(dst_policy: PPO, src_ckpt_dir: Path, init_std: float = 0.45):
    """将 18 维 Round 4 Actor 权重无损迁移到 19 维策略（索引 6 为新增 yaw_rad，初始化为 0）。"""
    src = _load_unpacked(src_ckpt_dir)
    src_sd = src.policy.state_dict()
    dst_sd = dst_policy.policy.state_dict()
    with torch.no_grad():
        for k, dst_t in dst_sd.items():
            if not (k.startswith("mlp_extractor.policy_net") or k.startswith("action_net")):
                continue
            src_t = src_sd[k]
            if src_t.shape == dst_t.shape:
                dst_sd[k] = src_t.clone()
            elif src_t.ndim == 2 and dst_t.ndim == 2 and src_t.shape[0] == dst_t.shape[0] and src_t.shape[1] + 1 == dst_t.shape[1]:
                new_w = torch.zeros_like(dst_t)
                new_w[:, :6] = src_t[:, :6]
                new_w[:, 6] = 0.0
                new_w[:, 7:] = src_t[:, 6:]
                dst_sd[k] = new_w
        dst_sd["log_std"] = torch.full_like(dst_sd["log_std"], float(np.log(init_std)))
    dst_policy.policy.load_state_dict(dst_sd)


class BestTrackerCallback(BaseCallback):
    def __init__(self, target_steps: int, best_dir: Path):
        super().__init__()
        self.target_steps = int(target_steps)
        self.best_dir = best_dir
        self.t0 = 0.0
        self.start_steps = 0
        self.iteration = 0
        self.best_rew = -1e9
        self.best_saved = False

    def _on_training_start(self):
        self.t0 = time.perf_counter()
        self.start_steps = self.num_timesteps

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.iteration += 1
        done = self.num_timesteps - self.start_steps
        elapsed = time.perf_counter() - self.t0
        fps = done / max(elapsed, 1e-6)
        buf = self.model.ep_info_buffer
        tag = ""
        if buf:
            mean_rew = float(np.mean([e["r"] for e in buf]))
            mean_len = float(np.mean([e["l"] for e in buf]))
            if self.iteration >= 2 and mean_len >= 320.0 and mean_rew > max(self.best_rew, 0.0):
                self.best_rew = mean_rew
                save_unpacked(self.model, self.best_dir)
                self.best_saved = True
                tag = " *BEST"
            print(f"iter={self.iteration:2d} | {done:7,d}/{self.target_steps:,d} | {fps:5.0f} step/s | rew={mean_rew:+6.3f} | len={mean_len:5.1f}{tag}", flush=True)
        else:
            print(f"iter={self.iteration:2d} | {done:7,d}/{self.target_steps:,d} | {fps:5.0f} step/s | waiting first ep", flush=True)


def main():
    parser = argparse.ArgumentParser(description="第 8 轮 MuJoCo Warp GPU 步态训练脚本（手动运行）。")
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=786432)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--warmstart-dir", type=Path, default=DEFAULT_WARMSTART_DIR)
    parser.add_argument("--resume-dir", type=Path, default=None, help="传入已展开的 19 维检查点目录以继续训练")
    parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "biped_gait_v8_out")
    args = parser.parse_args()

    env = VecMonitor(BipedGaitV8GpuVecEnv(n_envs=args.n_envs, episode_seconds=args.episode_seconds))
    out_dir = args.output_dir
    tb_dir = out_dir / "tb_logs"
    final_dir = out_dir / "final_checkpoint"
    best_dir = out_dir / "best_checkpoint"

    if args.resume_dir is not None:
        policy = _load_unpacked(args.resume_dir, custom_objects={"learning_rate": args.lr, "n_steps": args.n_steps})
        policy.set_env(env)
        policy.verbose = 0
        reset_steps = False
        print(f"续训 19 维检查点：{args.resume_dir}")
    else:
        policy = PPO(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=512,
            n_epochs=5,
            learning_rate=args.lr,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.005,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log=str(tb_dir),
            device="cpu",
            seed=42,
            verbose=0,
        )
        if args.warmstart_dir and args.warmstart_dir.exists():
            warmstart_actor_18_to_19(policy, args.warmstart_dir, init_std=0.45)
            print(f"已从 {args.warmstart_dir} 无损热启动 18->19 维 Actor。")
        reset_steps = True

    cb = BestTrackerCallback(args.total_timesteps, best_dir)
    policy.learn(total_timesteps=args.total_timesteps, tb_log_name="biped_gait_v8", reset_num_timesteps=reset_steps, callback=cb)
    save_unpacked(policy, final_dir)
    if cb.best_saved and best_dir.exists():
        print(f"最佳检查点（rew={cb.best_rew:+.3f}）已解包保存至：{best_dir}")
    print(f"最终检查点已解包保存至：{final_dir}")


if __name__ == "__main__":
    main()
