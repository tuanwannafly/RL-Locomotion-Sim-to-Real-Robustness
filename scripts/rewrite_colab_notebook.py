"""Rewrite colab/train_colab.ipynb with the fixes from the plan.

We construct the notebook as plain JSON so the result is well-formed and the
fixes (hardened GITHUB_REPO, robust token fetch, minimal Cell 4, single clean
TensorBoard cell, helpful verification cells) all take effect.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "colab" / "train_colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(*lines: str) -> dict:
    src = "\n".join(lines)
    if not src.endswith("\n"):
        src += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [src],
    }


cells = [
    md(
        "# RL Locomotion - Training on Google Colab\n"
        "\n"
        "PPO training of locomotion policies. This notebook runs the local "
        "`src/train.py` on a free Colab T4 GPU.\n"
        "\n"
        "## Configuration\n"
        "\n"
        "Edit the variables in the next cell, then `Runtime -> Run all`.\n"
    ),
    code(
        "# === EDIT ME ===",
        'GITHUB_REPO   = "https://github.com/tuanwannafly/RL-Locomotion-Sim-to-Real-Robustness.git"',
        'GITHUB_BRANCH = "main"',
        "",
        "try:",
        "    from google.colab import userdata",
        '    GITHUB_TOKEN = userdata.get("GITHUB_TOKEN") or ""',
        "except Exception:",
        '    GITHUB_TOKEN = ""   # public repo: leave empty if no secret set',
        "",
        'RUN_NAME    = "baseline_ant"',
        'CONFIG      = "configs/baseline_ant.yaml"',
        'ENV_ID      = "Ant-v4"',
        "TIMESTEPS   = 2_500_000",
        'DEVICE      = "cuda"',
        "",
        'DRIVE_RUNS  = "/content/drive/MyDrive/rl-locomotion/runs"',
        'REPO_DIR    = "/content/rl-locomotion"',
        'SAVE_DIR    = f"{DRIVE_RUNS}/{RUN_NAME}"',
        'print(f"Save dir: {SAVE_DIR}")',
    ),
    md("## 1. Mount Drive, clone repo, install deps"),
    code(
        "from google.colab import drive",
        'drive.mount("/content/drive")',
        "",
        "import os, subprocess, pathlib",
        "os.makedirs(DRIVE_RUNS, exist_ok=True)",
        'os.chdir("/content")',
        "",
        "if not pathlib.Path(REPO_DIR).exists():",
        "    url = GITHUB_REPO",
        "    if GITHUB_TOKEN:",
        '        url = url.replace("https://", f"https://{GITHUB_TOKEN}@")',
        '    subprocess.check_call(["git", "clone", "--branch", GITHUB_BRANCH, url, REPO_DIR])',
        "else:",
        '    subprocess.check_call(["git", "-C", REPO_DIR, "pull"])',
        "",
        "os.chdir(REPO_DIR)",
        'print("Repo at:", os.getcwd())',
        'print(subprocess.check_output(["git", "log", "--oneline", "-3"]).decode())',
    ),
    code(
        "# install PyTorch (CUDA) + project deps",
        'subprocess.check_call(["pip", "install", "-q",',
        '    "torch==2.7.0", "--index-url", "https://download.pytorch.org/whl/cu118"])',
        'subprocess.check_call(["pip", "install", "-q", "-r", "requirements.txt"])',
        'print("deps installed")',
    ),
    code(
        "# quick import sanity-check (catches broken installs before the long train)",
        "import gymnasium, stable_baselines3, torch, numpy",
        'print("gymnasium", gymnasium.__version__, "| sb3", stable_baselines3.__version__,',
        '      "| torch", torch.__version__, "| numpy", numpy.__version__)',
        'print("cuda available:", torch.cuda.is_available())',
    ),
    md("## 2. Verify GPU"),
    code(
        "!nvidia-smi -L || echo \"(no GPU)\"",
        "import torch",
        'print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")',
        "if torch.cuda.is_available():",
        '    print(f"device: {torch.cuda.get_device_name(0)}")',
        '    print(f"vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")',
    ),
    md(
        "## 3. Run training\n"
        "\n"
        "Calls `src/train.py` and streams logs into the notebook. Final model "
        "and TensorBoard events are saved to Drive."
    ),
    code(
        "import sys, subprocess",
        "sys.path.insert(0, REPO_DIR)",
        "from src.envs.path_utils import patch_gym_make",
        "patch_gym_make()",
        "",
        "cmd = [",
        '    "python", "src/train.py",',
        '    "--config",   CONFIG,',
        '    "--env",      ENV_ID,',
        '    "--timesteps", str(TIMESTEPS),',
        '    "--run-name", RUN_NAME,',
        '    "--save-dir", SAVE_DIR,',
        '    "--device",   DEVICE,',
        "]",
        'print(" ", " ".join(cmd))',
        "proc = subprocess.run(cmd, check=False)",
        'print(f"exit code: {proc.returncode}")',
    ),
    md("## 4. TensorBoard (optional)"),
    code(
        "%load_ext tensorboard",
        '%tensorboard --logdir "{DRIVE_RUNS}" --port 6006',
    ),
    md(
        "## 5. Download artefacts to local\n"
        "\n"
        "After the cell above finishes, go to "
        "`MyDrive/rl-locomotion/runs/<RUN_NAME>/` and download:\n"
        "\n"
        "- `model.zip`         -> local `experiments/models/<RUN_NAME>/model.zip`\n"
        "- `vecnormalize.pkl`  -> local `experiments/models/<RUN_NAME>/vecnormalize.pkl`\n"
        "- `monitor.csv`       -> local `experiments/logs/<RUN_NAME>/monitor.csv`\n"
        "- `tb/` (optional)    -> local `experiments/logs/<RUN_NAME>/tb/`\n"
        "\n"
        "Then continue with Sprint 1 analysis on local.\n"
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {NB_PATH} ({NB_PATH.stat().st_size} bytes, {len(cells)} cells)")
