"""Aggregate Sprint 2 ablation runs into a single CSV + comparison plot.

Usage:
    python src/summarize_ablation.py --runs-dir experiments/models --out-csv experiments/results/sprint2_ablation.csv --out-plot experiments/plots/sprint2_ablation.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402


def load_monitor(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "monitor.csv"
    if not p.exists():
        candidates = list(run_dir.glob("*.monitor.csv"))
        if not candidates:
            return None
        p = candidates[0]
    try:
        return pd.read_csv(p, skiprows=1)
    except Exception:
        return pd.read_csv(p)


def smooth(y: np.ndarray, window: int = 50) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    pad = window // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize ablation runs")
    p.add_argument("--runs-dir", default="experiments/models",
                   help="Directory containing run subdirs with monitor.csv")
    p.add_argument("--prefix", default="ablation",
                   help="Only include run dirs whose name starts with this prefix")
    p.add_argument("--out-csv", default="experiments/results/sprint2_ablation.csv")
    p.add_argument("--out-plot", default="experiments/plots/sprint2_ablation.png")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    runs_dir = (project_root / args.runs_dir).resolve()
    out_csv = (project_root / args.out_csv).resolve()
    out_plot = (project_root / args.out_plot).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([d for d in runs_dir.iterdir()
                       if d.is_dir() and d.name.startswith(args.prefix)])
    if not run_dirs:
        print(f"[ablation] no runs found under {runs_dir} with prefix {args.prefix!r}")
        return 1

    rows = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_dir in run_dirs:
        cfg_path = project_root / "configs" / f"{run_dir.name}.yaml"
        shaping = "?"
        lr = "?"
        gae = "?"
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            shaping = str(cfg.get("reward_shaping", {}))
            ppo = cfg.get("ppo", {})
            lr = ppo.get("learning_rate", "?")
            gae = ppo.get("gae_lambda", "?")

        df = load_monitor(run_dir)
        if df is None or len(df) == 0:
            print(f"[ablation] no monitor.csv in {run_dir}, skipping")
            continue
        rewards = df["r"].to_numpy()
        # final-100-ep mean as headline metric
        final_mean = float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards))
        best_mean = float(np.max(pd.Series(rewards).rolling(50, min_periods=1).mean()))
        rows.append({
            "run_name": run_dir.name,
            "reward_shaping": shaping,
            "learning_rate": lr,
            "gae_lambda": gae,
            "n_episodes": len(rewards),
            "mean_reward_last100": round(final_mean, 3),
            "best_smoothed_mean_reward": round(best_mean, 3),
        })
        smoothed = smooth(rewards, window=50)
        ax.plot(smoothed, label=run_dir.name)

    # write csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ablation] wrote {out_csv} ({len(rows)} rows)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed episode reward")
    ax.set_title("Sprint 2 - reward-shaping ablations")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    print(f"[ablation] wrote {out_plot}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())