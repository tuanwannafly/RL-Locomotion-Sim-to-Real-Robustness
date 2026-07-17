"""Plot robustness eval results.

Reads the CSV produced by ``src/eval.py`` and writes:
* a heatmap of mean_reward per (body_mass_scale, friction_scale) slice
* a comparison plot: bar chart of success_rate under nominal vs hardest perturbations

Usage:
    python src/plot_robustness.py --csv experiments/results/sprint4_robustness_ant.csv \
        --out-dir experiments/plots
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot robustness results")
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", default="experiments/plots")
    p.add_argument("--title", default=None)
    return p.parse_args()


def plot_heatmap(df: pd.DataFrame, policy: str, metric: str, value: str,
                 out_path: Path, title: str = "") -> None:
    sub = df[df["policy"] == policy]
    if len(sub) == 0:
        return
    pivot = sub.pivot_table(
        index="body_mass_scale",
        columns="friction_scale",
        values=value,
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{c:.2f}" for c in pivot.index])
    ax.set_xlabel("Friction scale")
    ax.set_ylabel("Body mass scale")
    ax.set_title(f"{title} {policy} - {metric} ({value})")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < np.nanmean(pivot.values) else "black",
                        fontsize=8)
    fig.colorbar(im, ax=ax, label=value)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_success_bars(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = df.pivot_table(
        index="body_mass_scale",
        columns="policy",
        values="success_rate",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Body mass scale")
    ax.set_title(f"{title} Success rate vs body mass")
    ax.legend(title="Policy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_delta_heatmap(df: pd.DataFrame, value: str, out_path: Path, title: str = "") -> None:
    base = df[df["policy"] == "baseline"].set_index(["body_mass_scale", "friction_scale"])[value]
    dr = df[df["policy"] == "dr"].set_index(["body_mass_scale", "friction_scale"])[value]
    delta = (dr - base).reset_index()
    pivot = delta.pivot_table(
        index="body_mass_scale",
        columns="friction_scale",
        values=value,
        aggfunc="mean",
    )
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", origin="lower",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{c:.2f}" for c in pivot.index])
    ax.set_xlabel("Friction scale")
    ax.set_ylabel("Body mass scale")
    ax.set_title(f"{title} DR advantage (delta {value})")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                sign = "+" if v >= 0 else ""
                ax.text(j, i, f"{sign}{v:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label=f"delta {value}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    csv_path = (project_root / args.csv).resolve() if not os.path.isabs(args.csv) else Path(args.csv)
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    title = args.title or csv_path.stem
    print(f"[plot] {len(df)} rows loaded from {csv_path}")

    for value, label in [("mean_reward", "reward"), ("success_rate", "success")]:
        for policy in ("baseline", "dr"):
            out = out_dir / f"{csv_path.stem}_heatmap_{policy}_{value}.png"
            plot_heatmap(df, policy, label, value, out, title=title)
        delta = out_dir / f"{csv_path.stem}_heatmap_delta_{value}.png"
        plot_delta_heatmap(df, value, delta, title=title)

    bars = out_dir / f"{csv_path.stem}_success_bars.png"
    plot_success_bars(df, bars, title=title)
    return 0


if __name__ == "__main__":
    sys.exit(main())