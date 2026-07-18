"""Run all tests in src/tests without requiring pytest.

Usage:
    .venv\\Scripts\\python.exe src/tests/run_all.py
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR.parent))


TEST_MODULES = [
    "src.tests.test_reward_wrappers",
    "src.tests.test_dr_and_action_delay",
    "src.tests.test_path_utils_and_eval_helpers",
]


def collect_tests(module):
    tests = []
    for name in dir(module):
        if name.startswith("test_") and callable(getattr(module, name)):
            tests.append(name)
    return tests


def main() -> int:
    n_pass, n_fail = 0, 0
    failed = []
    for mod_name in TEST_MODULES:
        module = importlib.import_module(mod_name)
        for test_name in collect_tests(module):
            full = f"{mod_name}.{test_name}"
            try:
                getattr(module, test_name)()
            except Exception:
                n_fail += 1
                failed.append((full, traceback.format_exc()))
                print(f"[FAIL] {full}")
            else:
                n_pass += 1
                print(f"[ ok ] {full}")

    total = n_pass + n_fail
    print(f"\n{n_pass}/{total} passed, {n_fail} failed")
    if failed:
        print("\n--- failure details ---")
        for name, tb in failed:
            print(f"\n>>> {name}\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())