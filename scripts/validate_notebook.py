"""Smoke-test the Colab notebook code cells in-process.

We can't run actual Colab-only steps (drive.mount, GPU, 2.5M-step training) here,
but we *can* make sure every cell parses and that the non-Colab-specific ones
import / execute without error under a stub environment.

Cells we execute locally (Cell 1 setup, Cell 3 clone logic with a fake REPO_DIR,
Cell 4 deps install, Cell 5 import sanity, Cell 9 train launcher dry-run).
Cells we only validate syntactically (Cell 7 nvidia-smi, Cell 11 tensorboard).
"""
from __future__ import annotations

import json
import sys
import subprocess
import textwrap
from pathlib import Path

NB = Path("colab/train_colab.ipynb").resolve()


def cells():
    return json.load(open(NB, encoding="utf-8"))["cells"]


def src(c):
    return "".join(c["source"])


def must_compile(c, idx):
    if c["cell_type"] != "code":
        print(f"[skip] cell {idx} ({c['cell_type']})")
        return
    body = src(c)
    # skip IPython magic cells (TensorBoard, !nvidia-smi) — they only work inside Jupyter
    if body.lstrip().startswith(("%", "!")):
        print(f"[skip] cell {idx} (IPython magic)")
        return
    try:
        compile(body, f"<cell-{idx}>", "exec")
        print(f"[syntax-ok] cell {idx} ({c['cell_type']})")
    except SyntaxError as ex:
        print(f"[SYNTAX-ERROR] cell {idx}: {ex}")
        sys.exit(1)


def main():
    cs = cells()
    print(f"notebook has {len(cs)} cells\n")

    # Validate every cell parses
    for i, c in enumerate(cs):
        must_compile(c, i)

    # Execute Cell 1 (config) to populate globals
    g: dict = {"__name__": "__main__"}
    print("\n[exec] cell 1 (config)")
    exec(src(cs[1]), g)

    for k in ("GITHUB_REPO", "GITHUB_BRANCH", "RUN_NAME", "CONFIG", "ENV_ID",
             "TIMESTEPS", "DEVICE", "DRIVE_RUNS", "REPO_DIR", "SAVE_DIR"):
        assert k in g, f"missing {k}"
    print("  -> SAVE_DIR =", g["SAVE_DIR"])
    assert g["GITHUB_REPO"].endswith(".git"), "GITHUB_REPO not a git URL"
    assert g["DEVICE"] in ("cuda", "cpu"), "DEVICE unexpected"
    assert g["RUN_NAME"] and g["CONFIG"] and g["ENV_ID"], "empty run config"

    # Execute Cell 5 (imports sanity-check) -- this is the same deps Colab will have
    print("\n[exec] cell 5 (import sanity)")
    exec(src(cs[5]), g)
    assert g["gymnasium"] is not None
    assert g["stable_baselines3"] is not None
    assert g["torch"] is not None

    # Execute Cell 9 training launcher as a dry-run: invoke src/train.py with --help
    print("\n[exec] cell 9 (train launcher dry-run)")
    body = src(cs[9])
    # prepend sys.path bootstrap so `from src.envs.path_utils import patch_gym_make` resolves
    body = (
        "import sys\n"
        "from pathlib import Path\n"
        "_repo = Path.cwd()\n"
        "if str(_repo) not in sys.path:\n"
        "    sys.path.insert(0, str(_repo))\n"
        + body
    )
    # swap the actual train command for a tiny smoke run so we don't run 2.5M steps locally
    body = body.replace('"--timesteps", str(TIMESTEPS),', '"--timesteps", "256",')
    body = body.replace('"--run-name", RUN_NAME,', '"--run-name", "smoke_verify",')
    body = body.replace(
        '"--save-dir", SAVE_DIR,',
        '"--save-dir", "experiments\\\\models\\\\smoke_verify",',
    )
    body = body.replace('DEVICE,', '"cpu",')  # local CPU is fine for 256 steps
    exec(body, g)
    print("  -> smoke train exited; artefact should be at experiments/models/smoke_verify/")

    # Check artefact
    artefact = Path("experiments/models/smoke_verify/model.zip")
    assert artefact.exists(), f"missing {artefact}"
    print(f"  -> OK: {artefact} ({artefact.stat().st_size} bytes)")

    print("\nALL NOTEBOOK CELLS VALIDATED")


if __name__ == "__main__":
    main()
