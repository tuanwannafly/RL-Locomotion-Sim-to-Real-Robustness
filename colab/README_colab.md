# Training on Google Colab

PPO training is too slow on the local RTX 2050 (4 GB VRAM, ~4-6 h per 2.5 M-step run).
This folder contains a Jupyter notebook that runs the same `src/train.py` on a free
Colab T4 GPU (~1.5-2.5 h per run, 16 GB VRAM, batch 256).

## Workflow

1. **Local** — code is committed to your repo. The notebook pulls the latest code at the top.
2. **Colab** — open `train_colab.ipynb`, set the run parameters at the top of the first cell, run all cells.
3. **Drive** — the notebook writes models, logs, and CSVs to `MyDrive/rl-locomotion/runs/<run_name>/`.
4. **Local** — once a run finishes, download the `.zip` and `monitor.csv` from Drive
   into `experiments/models/<run_name>/` and `experiments/logs/<run_name>/`. The
   agent will then plot training curves and run robustness eval.

## Setup on Colab (first run only)

1. Open `colab/train_colab.ipynb` in Colab.
2. In the first cell, set:
   - `GITHUB_REPO` — your remote Git URL (HTTPS).
   - `GITHUB_BRANCH` — usually `main` or `master`.
   - `GITHUB_TOKEN` — optional, only if the repo is private.
3. The cell will clone the repo at `/content/rl-locomotion`, mount Drive at
   `/content/drive/MyDrive/rl-locomotion`, and install dependencies.
4. The training cells call `python src/train.py --config <path> --env <id> ...`.
   Set `RUN_NAME`, `CONFIG`, `ENV_ID`, `TIMESTEPS` per run.

## Drive layout

```
MyDrive/rl-locomotion/
  runs/
    <run_name>/
      model.zip             # final SB3 model
      monitor.csv           # SB3 episode log (used by Sprint 4 plotting)
      vecnormalize.pkl      # VecNormalize stats (if used)
      tb/                   # TensorBoard event files
      training_log.txt      # stdout from training
```

## Syncing changes back to local

After Colab training finishes, do one of:

- **Manual**: download `model.zip` and `monitor.csv` from Drive into
  `experiments/models/<run_name>/` and `experiments/logs/<run_name>/`.
- **Drive desktop app**: sync the `rl-locomotion/runs/` folder.

The plotting and eval code expects models at `experiments/models/<run_name>/model.zip`
and logs at `experiments/logs/<run_name>/monitor.csv`.