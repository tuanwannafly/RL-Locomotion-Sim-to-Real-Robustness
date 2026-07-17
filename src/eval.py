"""Robustness evaluation harness.

Loads two trained policies (baseline and DR), evaluates each across a grid of
physics perturbations, and writes a CSV + summary plot.

Perturbations supported:
* body_mass_scale   (multiplicative on body mass)
* friction_scale    (multiplicative on dof_frictionloss)
* actuator_gain_scale (multiplicative on actuator_gear[:, 0])

The eval *applies* the perturbation directly to the underlying MuJoCo model
(no DR wrapper randomness); the same cell is deterministic for both policies.

Usage:
    python src/eval.py --env Ant-v4 \
        --baseline-run baseline_ant --dr-run dr_ant \
        --out-csv experiments/results/sprint4_robustness_ant.csv \
        --episodes-per-cell 30
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.path_utils import patch_gym_make
patch_gym_make()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from src.envs.reward_wrappers import make_shaped_env  # noqa: E402

DEFAULT_GRID = {
    "body_mass_scale":   [0.7, 0.85, 1.0, 1.15, 1.3],
    "friction_scale":    [0.5, 1.0, 1.5, 2.5],
    "actuator_gain_scale": [0.7, 1.0, 1.3],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness eval harness")
    p.add_argument("--env", required=True)
    p.add_argument("--baseline-run", required=True, help="Run dir with model.zip + (optional) vecnormalize.pkl")
    p.add_argument("--dr-run", required=True)
    p.add_argument("--episodes-per-cell", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--project-root", default=None)
    return p.parse_args()


def load_policy(run_dir: Path, env_id: str, device: str = "cpu"):
    model_path = run_dir / "model.zip"
    vecnorm_path = run_dir / "vecnormalize.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}")
    print(f"[eval] loading {model_path}")
    model = PPO.load(str(model_path), device=device)
    has_vecnorm = vecnorm_path.exists()
    return model, has_vecnorm, vecnorm_path


def apply_perturbation(env, body_mass_scale: float, friction_scale: float,
                       actuator_gain_scale: float) -> None:
    """Apply *fixed* multiplicative perturbations to the underlying MuJoCo model."""
    model = env.unwrapped.model
    if not hasattr(env, "_eval_default_body_mass"):
        env._eval_default_body_mass = model.body_mass.copy()
    if not hasattr(env, "_eval_default_dof_frictionloss"):
        env._eval_default_dof_frictionloss = model.dof_frictionloss.copy()
    if not hasattr(env, "_eval_default_actuator_gear"):
        env._eval_default_actuator_gear = model.actuator_gear.copy()
    model.body_mass[:] = env._eval_default_body_mass * body_mass_scale
    model.dof_frictionloss[:] = env._eval_default_dof_frictionloss * friction_scale
    model.actuator_gear[:, 0] = env._eval_default_actuator_gear[:, 0] * actuator_gain_scale


def rollout_episodes(model, env_id: str, perturbation: dict, n_episodes: int,
                     max_steps: int, has_vecnorm: bool, vecnorm_path: Path,
                     seed: int = 0) -> dict:
    """Run n_episodes with the given fixed perturbation, return aggregate metrics."""
    rewards = []
    ep_lens = []
    successes = 0  # survived >= 1000 steps

    # Build a base env (no DR), apply VecNormalize if available.
    def _make():
        e = make_shaped_env(env_id)
        return e

    base_vec = DummyVecEnv([_make])
    if has_vecnorm:
        venv = VecNormalize.load(str(vecnorm_path), base_vec)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = base_vec

    underlying = venv.venv.envs[0]  # actual gym env we mutate

    for ep in range(n_episodes):
        obs = venv.reset()
        ep_r = 0.0
        steps = 0
        for t in range(max_steps):
            # apply per-step perturbation (VecNormalize already done)
            apply_perturbation(
                underlying,
                perturbation.get("body_mass_scale", 1.0),
                perturbation.get("friction_scale", 1.0),
                perturbation.get("actuator_gain_scale", 1.0),
            )
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            ep_r += float(reward[0])
            steps += 1
            if done[0]:
                break
        rewards.append(ep_r)
        ep_lens.append(steps)
        if steps >= 1000:
            successes += 1

    venv.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_ep_len": float(np.mean(ep_lens)),
        "success_rate": successes / n_episodes,
        "n_episodes": n_episodes,
    }


def grid_iter(grid: dict) -> Iterable[dict]:
    keys = list(grid.keys())
    for combo in np.ndindex(*[len(grid[k]) for k in keys]):
        yield {k: grid[k][i] for k, i in zip(keys, combo)}


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root \
        else Path(__file__).resolve().parent.parent
    base_dir = project_root / "experiments" / "models"
    base_run = base_dir / args.baseline_run
    dr_run = base_dir / args.dr_run
    out_csv = (project_root / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] project_root={project_root}")
    print(f"[eval] baseline={base_run}, dr={dr_run}, episodes/cell={args.episodes_per_cell}")

    # load both
    base_model, base_vn, base_vn_path = load_policy(base_run, args.env)
    dr_model, dr_vn, dr_vn_path = load_policy(dr_run, args.env)

    rows = []
    grid = DEFAULT_GRID
    total_cells = int(np.prod([len(v) for v in grid.values()]))
    print(f"[eval] grid cells: {total_cells}")
    cell_idx = 0
    for pert in grid_iter(grid):
        cell_idx += 1
        for label, model, has_vn, vn_path in [
            ("baseline", base_model, base_vn, base_vn_path),
            ("dr",       dr_model,   dr_vn,   dr_vn_path),
        ]:
            print(f"[eval] cell {cell_idx}/{total_cells} {label} pert={pert}")
            metrics = rollout_episodes(
                model, args.env, pert, args.episodes_per_cell, args.max_steps,
                has_vn, vn_path, seed=cell_idx,
            )
            row = {"policy": label, **pert, **metrics}
            rows.append(row)
            print(f"         -> {metrics}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[eval] wrote {out_csv} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())