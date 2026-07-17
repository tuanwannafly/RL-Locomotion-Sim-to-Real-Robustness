# RL Locomotion — Sim-to-Real Robustness Project

> Train PPO policies for quadruped (Ant) and bipedal (Walker2d) locomotion in MuJoCo,
> and demonstrate that **domain randomization** measurably improves robustness to
> physics-perturbations — a controlled sim-to-real proxy without real hardware.

This README is a **stub**; full write-up is added in Sprint 6.

## Layout

```
src/                core code
  envs/             gymnasium wrappers (DR wrapper in Sprint 3)
  record_video.py   record one episode of any policy to mp4 (Sprint 0)
  train.py          PPO trainer (Sprint 1+)
  eval.py           robustness eval harness (Sprint 4)
  cross_engine_eval.py   MuJoCo -> PyBullet proxy (Sprint 5)
configs/            YAML hyperparameter configs per run
experiments/
  logs/             TensorBoard event files (gitignored)
  videos/           recorded episodes (gitignored)
  models/           saved SB3 .zip policies (gitignored)
  plots/            matplotlib outputs (gitignored)
  results/          CSV eval tables (gitignored)
colab/              training notebook for Google Colab
scripts/            one-shot .ps1 runners
```

## Quick start

```powershell
# 1. Create venv (Python 3.13 recommended)
uv venv --python 3.13 .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. CUDA-enabled PyTorch (RTX 2050 needs cu118 wheel)
uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118 --reinstall

# 4. Verify MuJoCo works
python src\record_video.py --env Ant-v4 --out experiments/videos/sprint0_random_ant.mp4 --max-steps 200
```

## Notes

- On Windows, paths containing non-ASCII characters trigger a MuJoCo
  parse error. `src/envs/path_utils.py` patches `gym.make` to redirect
  asset resolution to a temp directory whose path is ASCII-only.
  See comments in that file and MuJoCo issue
  [#3011](https://github.com/google-deepmind/mujoco/issues/3011).

## Sprint status

| Sprint | Title                            | Status     |
|--------|----------------------------------|------------|
| 0      | Setup + random-policy video      | Done       |
| 1      | Baseline PPO on Ant-v4 (Colab)   | upcoming   |
| 2      | Reward shaping & HP tuning       | upcoming   |
| 1b     | Baseline PPO on Walker2d-v4      | upcoming   |
| 3      | Domain randomization             | upcoming   |
| 4      | Robustness eval harness          | upcoming   |
| 5      | Cross-engine (PyBullet) eval     | upcoming   |
| 6      | README, video demo, packaging    | upcoming   |