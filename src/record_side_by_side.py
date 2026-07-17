"""Record a side-by-side comparison video of two policies under the same perturbation.

Renders both policies into separate mp4s, then concatenates them horizontally
into one side-by-side video using ffmpeg via moviepy.

Usage:
    python src/record_side_by_side.py \
        --baseline experiments/models/baseline_ant \
        --dr experiments/models/dr_ant \
        --env Ant-v4 --episodes 1 --max-steps 500 \
        --out experiments/videos/sprint6_ant_baseline_vs_dr.mp4
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.path_utils import patch_gym_make
patch_gym_make()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402


def record_single(model_dir: Path, env_id: str, out_mp4: Path,
                  episodes: int, max_steps: int, seed: int = 0) -> None:
    model = PPO.load(str(model_dir / "model.zip"), device="cpu")
    vecnorm = model_dir / "vecnormalize.pkl"
    has_vn = vecnorm.exists()

    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, str(out_mp4.parent),
                                   episode_trigger=lambda ep: ep < episodes,
                                   name_prefix=out_mp4.stem + "_part")

    base_vec = DummyVecEnv([lambda: gym.make(env_id)])
    if has_vn:
        venv = VecNormalize.load(str(vecnorm), base_vec)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = base_vec

    for ep in range(episodes):
        obs = venv.reset()
        env.reset(seed=seed + ep)
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            venv.step(action)
            env.step(action[0])  # for rendering only
            if hasattr(env, "should_record") and not env.should_record:
                break
            obs, _, done, _ = venv.step(action)
            if done[0]:
                break
    env.close()
    venv.close()


def combine_videos(left_mp4: Path, right_mp4: Path, out_mp4: Path) -> None:
    """Concatenate two mp4s horizontally using ffmpeg if available, else moviepy."""
    # try ffmpeg first (much faster, no re-encode artifacts)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-i", str(left_mp4), "-i", str(right_mp4),
               "-filter_complex", "hstack=inputs=2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)]
        print(f"[sxs] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return
        except subprocess.CalledProcessError as e:
            print(f"[sxs] ffmpeg failed ({e}), falling back to moviepy")
    # fallback: use moviepy to read both and stack
    from moviepy.editor import VideoFileClip, clips_array
    left = VideoFileClip(str(left_mp4))
    right = VideoFileClip(str(right_mp4))
    min_dur = min(left.duration, right.duration)
    left = left.subclip(0, min_dur)
    right = right.subclip(0, min_dur)
    combined = clips_array([[left, right]])
    combined.write_videofile(str(out_mp4), fps=30, codec="libx264", logger=None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side comparison video")
    p.add_argument("--baseline", required=True, help="Run dir of baseline policy")
    p.add_argument("--dr", required=True, help="Run dir of DR policy")
    p.add_argument("--env", required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output mp4 path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    baseline_dir = (project_root / args.baseline).resolve()
    dr_dir = (project_root / args.dr).resolve()
    out_path = (project_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_path.parent / "_sxs_tmp"
    tmp_dir.mkdir(exist_ok=True)
    left_mp4 = tmp_dir / "left.mp4"
    right_mp4 = tmp_dir / "right.mp4"
    try:
        print("[sxs] recording baseline")
        record_single(baseline_dir, args.env, left_mp4, args.episodes, args.max_steps, args.seed)
        print("[sxs] recording DR")
        record_single(dr_dir, args.env, right_mp4, args.episodes, args.max_steps, args.seed)
        print("[sxs] combining")
        combine_videos(left_mp4, right_mp4, out_path)
        print(f"[sxs] wrote {out_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())