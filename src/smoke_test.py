"""Smoke test: verify all core imports, CUDA, and MuJoCo envs work end-to-end.

Run with:
    .venv\\Scripts\\python.exe src\\smoke_test.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.path_utils import patch_gym_make
patch_gym_make()

import gymnasium as gym
import torch
import stable_baselines3
import numpy as np


def check(label: str, ok: bool, detail: str = "") -> None:
    status = "[OK]" if ok else "[FAIL]"
    print(f"{status} {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        sys.exit(1)


def main() -> None:
    print("=== RL Locomotion smoke test ===\n")

    # Versions
    print(f"python        {sys.version.split()[0]}")
    print(f"gymnasium     {gym.__version__}")
    print(f"stable_baselines3 {stable_baselines3.__version__}")
    print(f"torch         {torch.__version__}")
    print(f"numpy         {np.__version__}")
    print()

    # CUDA
    cuda = torch.cuda.is_available()
    print(f"cuda          available={cuda}")
    if cuda:
        print(f"              device={torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"              vram={vram_gb:.1f} GB")
    print()

    # Env: Ant-v4
    env = gym.make("Ant-v4")
    obs, info = env.reset(seed=0)
    check("Ant-v4 reset", obs.shape == env.observation_space.shape)
    for _ in range(10):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
    check("Ant-v4 step 10x", True, f"reward={r:.3f}")
    env.close()

    # Env: Walker2d-v4
    env = gym.make("Walker2d-v4")
    obs, info = env.reset(seed=0)
    check("Walker2d-v4 reset", obs.shape == env.observation_space.shape)
    for _ in range(10):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
    check("Walker2d-v4 step 10x", True, f"reward={r:.3f}")
    env.close()

    # SB3 PPO can be constructed
    from stable_baselines3 import PPO
    env = gym.make("Ant-v4")
    model = PPO("MlpPolicy", env, n_steps=128, batch_size=32, verbose=0, device="cpu")
    check("PPO MlpPolicy constructed", True, f"params={sum(p.numel() for p in model.policy.parameters())}")
    model.learn(total_timesteps=128)
    check("PPO learn(128 steps)", True)
    env.close()

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()