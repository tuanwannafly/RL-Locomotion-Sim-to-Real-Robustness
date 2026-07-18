"""Patch Cell 9 of colab/train_colab.ipynb so training runs inline (no subprocess).

Why:
  - On Colab Python 3.12, `subprocess.run(["python", "src/train.py", ...])` often
    segfaults at import time because the spawned `python` may resolve to a
    different interpreter than `sys.executable` of the kernel, and because
    `jupyter_client`'s pyzmq + mujoco's libomp sometimes collide on the
    Resolver/Address stack.
  - Running `src.train.main()` directly inside the kernel bypasses both issues
    and gives us a single process to attach a traceback to.

The new Cell 9:
  - Sets OMP_NUM_THREADS=1 and KMP_DUPLICATE_LIB_OK=TRUE so libomp doesn't
    double-load inside torch + mujoco.
  - Imports `src.train.main` lazily and calls it with the same argv-style args
    the subprocess would have used.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path("colab/train_colab.ipynb").resolve()
nb = json.load(open(NB, encoding="utf-8"))


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


new_cell_9 = code(
    "# Workaround: avoid native-lib segfaults seen when running train.py via",
    "# `subprocess.run([\"python\", ...])` on Colab Python 3.12 (libomp/zmq).",
    "import os",
    'os.environ.setdefault("OMP_NUM_THREADS", "1")',
    'os.environ.setdefault("MKL_NUM_THREADS", "1")',
    'os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")',
    "",
    "import sys",
    "sys.path.insert(0, REPO_DIR)",
    "os.chdir(REPO_DIR)",
    "",
    "# Import lazily so the kernel state (Drive mount, clone) is ready first.",
    "from src.train import main as train_main",
    "",
    "# Drive argv the way src/train.py expects it.",
    "sys.argv = [",
    '    "src/train.py",',
    '    "--config",   CONFIG,',
    '    "--env",      ENV_ID,',
    '    "--timesteps", str(TIMESTEPS),',
    '    "--run-name", RUN_NAME,',
    '    "--save-dir", SAVE_DIR,',
    '    "--device",   DEVICE,',
    "]",
    "",
    "# Streaming a long train (2.5M steps) live in the cell output is hard;",
    "# if it dies early you will see a real Python traceback (not -11) now.",
    "raise SystemExit(train_main())",
)

# The original Cell 9 was the subprocess-launcher. Replace it.
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if c["cell_type"] == "code" and src.lstrip().startswith("import sys, subprocess"):
        nb["cells"][i] = new_cell_9
        print(f"replaced Cell {i}")
        break
else:
    raise RuntimeError("Could not find original Cell 9 (subprocess launcher)")

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("done")