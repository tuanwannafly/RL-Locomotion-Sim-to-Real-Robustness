"""Sprint 1+ trainer.

Trains an SB3 PPO policy on a Gymnasium MuJoCo env. Driven by a YAML config
file. Designed to run both locally (RTX 2050) and on Google Colab (T4 16 GB).

CLI usage:
    python src/train.py --config configs/baseline_ant.yaml --env Ant-v4 \
        --timesteps 2500000 --run-name baseline_ant --device cuda

The script auto-applies the MuJoCo Windows path workaround before importing
gymnasium mujoco envs, so paths with non-ASCII characters (this workspace's
em-dash) don't break asset resolution.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Path workaround must run BEFORE gymnasium mujoco envs are imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.path_utils import patch_gym_make  # noqa: E402
patch_gym_make()

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from src.envs.reward_wrappers import make_shaped_env  # noqa: E402
from src.envs.dr_wrapper import make_dr_env  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SB3 PPO on a Gymnasium MuJoCo env")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--env", default=None, help="Override env id from config")
    p.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps")
    p.add_argument("--run-name", default=None, help="Run name (default: derive from config)")
    p.add_argument("--save-dir", default=None,
                   help="Override save dir (default: experiments/models/<run-name>)")
    p.add_argument("--device", default=None, help='Override device: "cuda" or "cpu"')
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    p.add_argument("--no-vecnorm", action="store_true",
                   help="Disable VecNormalize wrapper (default: enabled)")
    return p.parse_args()


def merge_config(args: argparse.Namespace) -> tuple[dict, str, int, str, str, int, bool]:
    """Resolve CLI overrides onto the YAML config.

    Returns: (cfg, env_id, timesteps, run_name, save_dir, seed, vecnorm_enabled)
    """
    cfg = load_config(args.config)
    env_id = args.env or cfg.get("env_id", "Ant-v4")
    timesteps = args.timesteps or int(cfg.get("training", {}).get("total_timesteps", 1_000_000))

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = cfg.get("run_name") or Path(args.config).stem

    project_root = Path(__file__).resolve().parent.parent
    save_dir = args.save_dir or str(project_root / "experiments" / "models" / run_name)

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    vecnorm = (not args.no_vecnorm) and bool(cfg.get("training", {}).get("vec_normalize", True))
    return cfg, env_id, timesteps, run_name, save_dir, seed, vecnorm


def make_env_fn(env_id: str, seed: int, monitor_dir: str | None = None,
                shaping_cfg: dict | None = None, dr_cfg: dict | None = None):
    """Return a thunk that constructs a Monitor-wrapped env with reward shaping
    and optional domain randomization.

    If *monitor_dir* is given, the Monitor wrapper writes ``<monitor_dir>/<env_id>.monitor.csv``
    so the path is reproducible across runs.
    """
    def _thunk():
        env = make_dr_env(env_id, dr_cfg=dr_cfg, seed=seed, shaping_cfg=shaping_cfg)
        if monitor_dir is not None:
            os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=os.path.join(monitor_dir, env_id))
        else:
            env = Monitor(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _thunk


def main() -> int:
    args = parse_args()
    cfg, env_id, timesteps, run_name, save_dir, seed, vecnorm = merge_config(args)
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    ppo_cfg = cfg.get("ppo", {})
    train_cfg = cfg.get("training", {})
    log_cfg = cfg.get("logging", {})

    # Build env (single-env wrapped in DummyVecEnv -> VecNormalize optional)
    n_envs = int(train_cfg.get("n_envs", 1))
    monitor_dir = str(Path(__file__).resolve().parent.parent / "experiments" / "logs" / run_name)
    shaping_cfg = cfg.get("reward_shaping", {}) or {}
    dr_cfg = cfg.get("domain_randomization", {}) or {}
    env_fns = [make_env_fn(env_id, seed + i, monitor_dir=monitor_dir if i == 0 else None,
                           shaping_cfg=shaping_cfg, dr_cfg=dr_cfg)
               for i in range(n_envs)]
    if n_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        # SB3 exposes SubprocVecEnv, but for Sprint 1 we keep things simple.
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv(env_fns)
    if vecnorm:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=ppo_cfg.get("gamma", 0.99))

    # TensorBoard log dir; SB3 creates the dir itself.
    tb_root = log_cfg.get("tensorboard_log", "experiments/logs")
    tb_dir = Path(tb_root) / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_root = str(tb_dir.parent)  # SB3 expects the parent of the run folder
    run_tb_dir = str(tb_dir)

    # Checkpoints during training (also final .zip saved via .save()).
    save_freq = int(log_cfg.get("save_freq", 50_000))
    ckpt_callback = CheckpointCallback(
        save_freq=max(save_freq // max(n_envs, 1), 1),
        save_path=str(save_dir_path / "ckpt"),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    device = args.device or train_cfg.get("device", "cuda")
    print(f"[train] env={env_id}  timesteps={timesteps}  run={run_name}  device={device}")
    print(f"[train] save_dir={save_dir_path}  vecnorm={vecnorm}  n_envs={n_envs}")
    print(f"[train] reward_shaping={shaping_cfg}")
    print(f"[train] domain_randomization={ {k: v for k, v in dr_cfg.items() if k != 'enabled'} } (enabled={dr_cfg.get('enabled', False)})")
    print(f"[train] ppo={ppo_cfg}")

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=__import__("torch.nn", fromlist=["ReLU"]).ReLU,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 256),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.0),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tb_root,
        seed=seed,
        device=device,
    )
    # SB3 expects the run name to be supplied via a kwarg on .learn()
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=timesteps,
            tb_log_name=run_name,
            callback=ckpt_callback,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("[train] interrupted, saving partial model")
    elapsed = time.time() - t0

    # Final artefacts.
    model.save(str(save_dir_path / "model"))
    if vecnorm:
        env.save(str(save_dir_path / "vecnormalize.pkl"))
    # Save monitor.csv if Monitor wrote one (sb3 Monitor appends .monitor.csv).
    monitor_path = Path(monitor_dir) / f"{env_id}.monitor.csv"
    if monitor_path.exists():
        dst = save_dir_path / "monitor.csv"
        dst.write_bytes(monitor_path.read_bytes())
        print(f"[train] copied monitor.csv -> {dst}")
    print(f"[train] done in {elapsed/60:.1f} min. saved to {save_dir_path}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())