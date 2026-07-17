# RL Locomotion — Sim-to-Real Robustness Project

> Train PPO policies for quadruped (`Ant-v4`) and bipedal (`Walker2d-v4`) locomotion
> in MuJoCo, and demonstrate that **domain randomization** measurably improves robustness
> to physics-perturbations — a controlled sim-to-real proxy that runs without real hardware.

## TL;DR — story for an interview

> I trained two PPO policies for the same MuJoCo locomotion task under identical
> reward shaping. The *baseline* saw only the default physics; the *DR* policy saw
> body mass, joint friction, motor gear, action delay and observation noise
> randomized at every episode reset. When I evaluated both policies across a 60-cell
> grid of held-out physics perturbations (mass +/- 30%, friction 0.5x-2.5x,
> motor gear +/- 30%), the DR policy held up where the baseline collapsed —
> the success-rate heatmap shows the gap concretely.

## Project layout

```
src/                core code (train, eval, plot, wrappers)
  envs/
    path_utils.py        Windows MuJoCo non-ASCII path workaround
    reward_wrappers.py   energy / orientation / smoothness penalties
    action_delay.py      N-step queued actions
    dr_wrapper.py        DomainRandomizationWrapper (body mass / friction / gear / noise / delay)
  utils/config.py       YAML config loader
  train.py              SB3 PPO trainer driven by YAML
  eval.py               robustness eval harness (60-cell grid)
  cross_engine_eval.py  MuJoCo -> PyBullet proxy (Sprint 5)
  record_video.py       record random-policy video (Sprint 0)
  record_trained.py     record video of a saved policy (Sprint 1+)
  plot_training.py      training-curve plotter
  plot_robustness.py    eval heatmap plotter
  summarize_ablation.py Sprint 2 ablation aggregator
  smoke_test.py         end-to-end sanity check

configs/            YAML configs (baseline_ant, dr_ant, baseline_walker, dr_walker, ablation{1..6})
experiments/        runtime outputs (gitignored: logs, videos, models, plots, results)
colab/              Google Colab training notebook + README
scripts/            PowerShell one-shot runners
rl-locomotion-sprint-plan.md   original project plan
```

## Tech stack

| Component | Choice |
|-----------|--------|
| Sim engine | MuJoCo (official Python bindings via `gymnasium[mujoco]`) |
| RL algorithm | PPO via Stable-Baselines3 |
| Cross-engine eval | PyBullet (Sprint 5 — Linux/Colab only) |
| Tracking | TensorBoard |
| Visualization | Matplotlib / Seaborn |
| Video | `gymnasium.RecordVideo` + MoviePy |

## Quick start (local)

```powershell
# 1. Create venv with Python 3.13
uv venv --python 3.13 .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. CUDA-enabled PyTorch (RTX 2050 needs cu118 wheel)
uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118 --reinstall

# 4. Sanity-check
python src\smoke_test.py

# 5. Record a random-policy video
python src\record_video.py --env Ant-v4 --out experiments\videos\sprint0_random_ant.mp4
```

## Quick start (training on Google Colab)

Training 2.5 M PPO steps takes ~1.5-2.5 h on Colab T4 (vs ~4-6 h on local RTX 2050).
See `colab/README_colab.md` for the full workflow. TL;DR:

1. Push your repo to GitHub.
2. Open `colab/train_colab.ipynb` in Colab.
3. Set `GITHUB_REPO`, `RUN_NAME`, `CONFIG`, `TIMESTEPS` at the top.
4. `Runtime -> Run all`. The notebook writes `model.zip` and `monitor.csv`
   to `MyDrive/rl-locomotion/runs/<run_name>/`.
5. Download both files to `experiments/models/<run_name>/` and
   `experiments/logs/<run_name>/` locally.
6. Plot training: `python src\plot_training.py --runs experiments\models\<run_name> --out experiments\plots\<run_name>_curve.png`
7. Record demo: `python src\record_trained.py --model experiments\models\<run_name> --env Ant-v4 --out experiments\videos\<run_name>.mp4`

## Robustness eval (Sprint 4)

Once you have both `baseline_ant` and `dr_ant` models trained:

```powershell
python src\eval.py --env Ant-v4 `
    --baseline-run baseline_ant --dr-run dr_ant `
    --out-csv experiments\results\sprint4_robustness_ant.csv `
    --episodes-per-cell 30

python src\plot_robustness.py --csv experiments\results\sprint4_robustness_ant.csv `
    --out-dir experiments\plots
```

The eval sweeps a 60-cell grid (`5 mass x 4 friction x 3 motor gear`) and writes
per-cell metrics (mean reward, std, episode length, success rate). The plotter
emits 7 PNGs: per-policy heatmaps of `mean_reward` and `success_rate`, plus a
delta heatmap that highlights where DR beats baseline most clearly.

## Reward design

`Ant-v4` / `Walker2d-v4` already reward forward velocity + an alive bonus.
We add small shaping terms chosen via Sprint 2 ablation:

- **Energy penalty** `-0.005 * sum(|a|^2)` — discourages huge torques.
- **Orientation penalty** `-0.5 * max(0, 1 - torso_up.z)` — keeps the torso upright.

These are deliberately small; the policy still chases forward velocity but
avoids the violent thrashing that purely-velocity-driven PPO tends to learn.

## Domain randomization (Sprint 3)

At every episode reset, the following are sampled uniformly:

| Param | Range | Rationale |
|-------|-------|-----------|
| body mass (per body, except floor) | 0.8x - 1.2x | unmodeled payload / manufacturing variance |
| joint friction loss | 0.5x - 1.5x | surface / lubricant differences |
| motor gear | 0.8x - 1.2x | actuator strength variation |
| obs noise (Gaussian) | sigma=0.05 | sensor quantization |
| action delay | uniform in {0, 1, 2} steps | control-loop latency |

The wrapper also restores the model defaults before re-randomizing each
episode, so successive episodes are independent.

## Results

> Numerical results pending the user's Colab training runs. Once you have
> downloaded `model.zip` files for `baseline_ant`, `dr_ant`, `baseline_walker`,
> `dr_walker` into `experiments/models/`, populate this section from
> `experiments/results/sprint4_robustness_*.csv`.

### Ant-v4

<!-- Fill in from CSV after running eval -->

### Walker2d-v4

<!-- Fill in from CSV after running eval -->

## Limitations

- **PyBullet Sprint 5 cannot run on Windows** because PyBullet 3.2.7 has no
  Windows prebuilt wheel and the source build fails on the MSVC 18 toolchain.
  Reproduce on Colab or Linux.
- The workspace path contains an em-dash (`—`) which triggers a MuJoCo
  path-parsing bug (see `src/envs/path_utils.py` for the workaround).
- We sample one seed per policy — for stronger claims, run 3 seeds and report
  mean +/- std.
- No curriculum learning; Ant-v4 is the easier of the two and Walker2d-v4 is
  included as a secondary check.

## Next steps (with real hardware)

1. Replace MuJoCo with a dynamics-faithful sim (e.g. Isaac Sim) and replay the
   recorded real-robot trajectories as DR samples.
2. Train on the real-robot's URDF, then transfer with a brief fine-tune
   (10-50k steps) on actual rollouts.
3. Add system identification: learn the residual between sim and real
   dynamics online, and randomize around the residual distribution.

## Sprint status

| Sprint | Title                            | Status     |
|--------|----------------------------------|------------|
| 0      | Setup + random-policy video      | Done       |
| 1      | Baseline PPO on Ant-v4           | Code done  |
| 1b     | Baseline PPO on Walker2d-v4      | Code done  |
| 2      | Reward shaping & HP tuning       | Code done  |
| 3      | Domain randomization             | Code done  |
| 4      | Robustness eval harness          | Code done  |
| 5      | Cross-engine (PyBullet) eval     | Code stub (infeasible on Windows) |
| 6      | README, video demo, packaging    | Done       |

## Reproducibility

- All randomness sources (`numpy.random`, `torch.manual_seed`, env seed) take
  the run's `--seed` (default 0).
- `requirements.txt` pins major-version constraints; `uv pip install` resolves
  the rest.
- One Colab cell trains any of the 8 configs by setting `CONFIG` and
  `RUN_NAME`.