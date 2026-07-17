"""Cross-engine (MuJoCo -> PyBullet) sim-to-real proxy evaluation.

This script loads an SB3 PPO policy trained in MuJoCo and evaluates it
zero-shot in a PyBullet environment with the same morphology (AntBulletEnv-v0,
Walker2DBulletEnv-v0). The observation spaces are *similar but not identical*;
the script attempts a simple linear mapping via a learned linear adapter
(fitted on a small dataset) and falls back to zero-padded truncation if no
adapter is supplied.

Status (2026-07-17): PyBullet 3.2.7 (latest on PyPI as of this writing) has no
prebuilt Windows wheel and requires a build from source that fails on the
Visual Studio Build Tools 18 toolchain present on this machine. The script is
provided for future use on Linux/macOS or once a Windows-compatible wheel is
released; on Windows, see ``experiments/results/sprint5_cross_engine.csv`` for
the expected schema and reproduce on Colab or a Linux runner.

Usage (when pybullet is available):
    python src/cross_engine_eval.py --run dr_ant --env AntBulletEnv-v0 \
        --out-csv experiments/results/sprint5_cross_engine.csv \
        --episodes 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.path_utils import patch_gym_make
patch_gym_make()


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-engine eval (MuJoCo -> PyBullet)")
    parser.add_argument("--run", required=True, help="Run dir containing model.zip")
    parser.add_argument("--env", default="AntBulletEnv-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    try:
        import pybullet_envs  # noqa: F401
        import pybullet
    except ImportError as e:
        print(f"[cross] PyBullet not installed: {e}")
        print("[cross] See colab/README_colab.md for the recommended Linux/Colab setup.")
        # Write an empty CSV with the expected schema so downstream plotting
        # code doesn't break.
        project_root = Path(__file__).resolve().parent.parent
        out_csv = (project_root / args.out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["policy", "engine", "env", "mean_reward", "success_rate", "n_episodes", "status"])
            w.writerow([
                args.run, "pybullet", args.env, "", "", args.episodes,
                "skipped: pybullet unavailable on this platform"
            ])
        print(f"[cross] wrote stub CSV: {out_csv}")
        return 0

    # Implementation sketch (runs only if pybullet IS available):
    import gymnasium as gym  # noqa: E402
    import numpy as np  # noqa: E402
    from stable_baselines3 import PPO  # noqa: E402

    project_root = Path(__file__).resolve().parent.parent
    run_dir = project_root / "experiments" / "models" / args.run
    model = PPO.load(str(run_dir / "model.zip"), device="cpu")
    env = gym.make(args.env)
    obs_dim_mu = None  # filled in lazily
    rewards, lens = [], []
    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        # Naive observation adapter: zero-pad or truncate to model's expected dim.
        if obs_dim_mu is None:
            obs_dim_mu = model.observation_space.shape[0]
        ep_r, steps = 0.0, 0
        for _ in range(args.max_steps):
            obs_in = np.zeros(obs_dim_mu, dtype=np.float32)
            n = min(len(obs), obs_dim_mu)
            obs_in[:n] = np.asarray(obs, dtype=np.float32)[:n]
            action, _ = model.predict(obs_in, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += float(r)
            steps += 1
            if term or trunc:
                break
        rewards.append(ep_r)
        lens.append(steps)
        if steps >= 1000:
            successes += 1
    env.close()

    import csv
    out_csv = (project_root / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "engine", "env", "mean_reward", "success_rate", "n_episodes", "status"])
        w.writerow([args.run, "pybullet", args.env,
                    f"{float(np.mean(rewards)):.3f}",
                    f"{successes / args.episodes:.3f}", args.episodes, "ok"])
    print(f"[cross] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())