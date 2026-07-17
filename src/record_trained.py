"""Record a video of a trained SB3 PPO policy.

Usage:
    python src/record_trained.py --model experiments/models/baseline_ant/model.zip \
        --env Ant-v4 --out experiments/videos/baseline_ant.mp4 --episodes 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.path_utils import patch_gym_make
patch_gym_make()

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record trained SB3 policy")
    p.add_argument("--model", required=True, help="Path to model.zip (or run dir)")
    p.add_argument("--env", required=True, help="Gymnasium env id")
    p.add_argument("--out", required=True, help="Output mp4 path (project-relative)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-vecnorm", action="store_true", help="Skip VecNormalize load")
    return p.parse_args()


def resolve_model_path(model_arg: str) -> Path:
    p = Path(model_arg)
    if p.is_dir():
        p = p / "model.zip"
    return p.resolve()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    out_path = (project_root / args.out).resolve() if not os.path.isabs(args.out) else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.model)
    print(f"[record] loading model from {model_path}")
    model = PPO.load(str(model_path), device="cpu")

    # Try to load matching VecNormalize if present (same dir as model.zip)
    vecnorm_path = model_path.parent / "vecnormalize.pkl"
    has_vecnorm = vecnorm_path.exists() and not args.no_vecnorm

    env = gym.make(args.env, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        str(out_path.parent),
        episode_trigger=lambda ep: ep < args.episodes,
        name_prefix=out_path.stem,
    )
    env.reset(seed=args.seed)

    # When the policy was trained with VecNormalize, we must wrap & load stats
    # before predicting so the obs is normalized the same way. For inference
    # recording, we apply VecNormalize as a wrapper that *normalizes* but
    # doesn't clip rewards.
    if has_vecnorm:
        venv = DummyVecEnv([lambda: gym.make(args.env)])
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
        # Use the loaded vecnorm for prediction, but render from the original env
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            for _ in range(args.max_steps):
                obs_norm = venv.normalize_obs(obs[None, :])
                action, _ = model.predict(obs_norm, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action[0])
                if terminated or truncated:
                    break
    else:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            for _ in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

    env.close()

    expected = out_path.parent / f"{out_path.stem}-episode-0.mp4"
    if expected.exists() and expected != out_path:
        # there may be multiple episode files; keep the first as the canonical
        # rename and remove the rest
        for ep in range(args.episodes):
            ep_file = out_path.parent / f"{out_path.stem}-episode-{ep}.mp4"
            if ep_file.exists() and ep_file == expected:
                ep_file.replace(out_path)
            elif ep_file.exists():
                ep_file.unlink()
    print(f"[record] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())