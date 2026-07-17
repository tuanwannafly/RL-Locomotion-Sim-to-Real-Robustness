"""Plot training curves from SB3 monitor.csv files.

Usage:
    python src/plot_training.py --runs experiments/models/baseline_ant --out experiments/plots/baseline_ant_curve.png
    python src/plot_training.py --runs experiments/models/baseline_ant experiments/models/baseline_walker --labels Ant Walker --out experiments/plots/comparison.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def smooth(y: np.ndarray, window: int = 50) -> np.ndarray:
    """Centered moving-average smoothing."""
    if window <= 1 or len(y) < window:
        return y
    pad = window // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def load_monitor(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "monitor.csv"
    if not p.exists():
        # fallback: search any *.monitor.csv
        candidates = list(run_dir.glob("*.monitor.csv"))
        if not candidates:
            return None
        p = candidates[0]
    # SB3 Monitor writes a 2-line header before the data CSV.
    try:
        return pd.read_csv(p, skiprows=1)
    except Exception:
        return pd.read_csv(p)


def plot_runs(run_dirs: list[Path], labels: list[str], out_path: Path,
              metric: str = "r", window: int = 50) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_dir, label in zip(run_dirs, labels):
        df = load_monitor(run_dir)
        if df is None or len(df) == 0:
            print(f"[plot] no monitor.csv in {run_dir}, skipping")
            continue
        # SB3 columns: r (episode reward), l (episode length), t (wall time)
        col = metric if metric in df.columns else "r"
        y = df[col].to_numpy()
        x = np.arange(len(y))
        ax.plot(x, smooth(y, window), label=f"{label} (smoothed)")
        ax.plot(x, y, alpha=0.2, linewidth=0.5, color=ax.lines[-1].get_color())
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward" if metric == "r" else metric)
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot] wrote {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot SB3 training curves")
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more run dirs containing monitor.csv")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Legend labels (one per run, default: dir name)")
    p.add_argument("--metric", default="r", choices=["r", "l", "t"])
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--out", required=True, help="Output png path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    run_dirs = [Path(r).resolve() for r in args.runs]
    labels = args.labels or [d.name for d in run_dirs]
    if len(labels) != len(run_dirs):
        print("[plot] --labels count must match --runs count")
        return 1
    out_path = (project_root / args.out).resolve() if not os.path.isabs(args.out) else Path(args.out)
    plot_runs(run_dirs, labels, out_path, args.metric, args.window)
    return 0


if __name__ == "__main__":
    sys.exit(main())