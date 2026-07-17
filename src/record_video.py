"""Sprint 0 - record one episode of a random policy to verify MuJoCo + Gymnasium work.

Usage:
    .venv\\Scripts\\python.exe src\\record_video.py --env Ant-v4 --out experiments/videos/sprint0_random_ant.mp4
    .venv\\Scripts\\python.exe src\\record_video.py --env Walker2d-v4 --out experiments/videos/sprint0_random_walker.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Apply Windows MuJoCo path workaround BEFORE importing gymnasium mujoco envs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.path_utils import patch_gym_make
patch_gym_make()

import gymnasium as gym


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record one episode of a random policy")
    p.add_argument("--env", default="Ant-v4", help="Gymnasium env id (default: Ant-v4)")
    p.add_argument("--out", default="experiments/videos/sprint0_random.mp4",
                   help="Output mp4 path (relative to project root)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500,
                   help="Episode length cap (Ant-v4 default is 1000)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # resolve output path relative to project root (parent of src/)
    project_root = Path(__file__).resolve().parent.parent
    out_path = (project_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = out_path.parent
    out_name = out_path.stem  # name without extension

    # Make env with rgb_array renderer, wrap with RecordVideo.
    env = gym.make(args.env, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        str(out_dir),
        episode_trigger=lambda ep: ep == 0,  # only record episode 0
        name_prefix=out_name,
    )

    obs, info = env.reset(seed=args.seed)
    for _ in range(args.max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()

    # RecordVideo writes "<name_prefix>-episode-<idx>.mp4" to out_dir
    expected = out_dir / f"{out_name}-episode-0.mp4"
    if expected.exists():
        if expected != out_path:
            expected.replace(out_path)
        print(f"[OK] Saved video to {out_path}")
    else:
        print(f"[WARN] Expected file {expected} not found.")
        # List whatever was written
        for f in out_dir.glob(f"{out_name}*"):
            print(f"  found: {f}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())