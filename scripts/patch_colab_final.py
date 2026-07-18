import json, sys

nb = json.load(open("colab/train_colab.ipynb", encoding="utf-8"))

# ---------- Cell 9: add Ant-v4 sanity + fix tb logdir ----------
sanity_lines = [
    "# Quick sanity: verify Ant-v4 env can step before the long train.\n",
    "import gymnasium as gym\n",
    'print("Sanity: Ant-v4 reset/step...", end=" ", flush=True)\n',
    '_test_env = gym.make("Ant-v4")\n',
    "_obs, _info = _test_env.reset()\n",
    "for _ in range(10):\n",
    "    _a = _test_env.action_space.sample()\n",
    "    _obs, _r, _t, _tr, _i = _test_env.step(_a)\n",
    "    if _t or _tr: _test_env.reset()\n",
    "_test_env.close()\n",
    'print("OK")\n',
    "\n",
]

cell9 = nb["cells"][9]
cell9["source"] = (
    [
        "# Workaround: avoid native-lib segfaults seen when running train.py via\n",
        '# `subprocess.run(["python", ...])` on Colab Python 3.12 (libomp/zmq).\n',
        "import os\n",
        'os.environ.setdefault("OMP_NUM_THREADS", "1")\n',
        'os.environ.setdefault("MKL_NUM_THREADS", "1")\n',
        'os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")\n',
        "\n",
        "import sys\n",
        "sys.path.insert(0, REPO_DIR)\n",
        "os.chdir(REPO_DIR)\n",
        "\n",
    ]
    + sanity_lines
    + [
        "# Import lazily so the kernel state (Drive mount, clone) is ready first.\n",
        "from src.train import main as train_main\n",
        "\n",
        "# Drive argv the way src/train.py expects it.\n",
        "sys.argv = [\n",
        '    "src/train.py",\n',
        "    \"--config\",   CONFIG,\n",
        "    \"--env\",      ENV_ID,\n",
        '    "--timesteps", str(TIMESTEPS),\n',
        '    "--run-name", RUN_NAME,\n',
        '    "--save-dir", SAVE_DIR,\n',
        '    "--device",   DEVICE,\n',
        "]\n",
        "\n",
        "# Streaming a long train (2.5M steps) live in the cell output is hard;\n",
        "# if it dies early you will see a real Python traceback (not -11) now.\n",
        "raise SystemExit(train_main())\n",
    ]
)

# ---------- Cell 11: fix TensorBoard logdir to SAVE_DIR ----------
nb["cells"][11]["source"] = [
    "%load_ext tensorboard\n",
    '%tensorboard --logdir "{SAVE_DIR}" --port 6006\n',
]

json.dump(nb, open("colab/train_colab.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("done")
