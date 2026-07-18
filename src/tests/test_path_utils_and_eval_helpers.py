"""Tests for path_utils (Windows MuJoCo workaround)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_short_path_round_trip():
    """Any existing path should short-circuit to itself on non-Windows or to a
    gettable short path on Windows."""
    from src.envs.path_utils import short_path
    p = os.path.abspath(".")
    sp = short_path(p)
    assert isinstance(sp, str)
    assert os.path.exists(sp)


def test_short_path_missing_returns_input():
    from src.envs.path_utils import short_path
    assert short_path("/nonexistent/path/xyz") == "/nonexistent/path/xyz"


def test_patch_gym_make_is_idempotent_and_works():
    """patch_gym_make must be safe to call multiple times."""
    from src.envs.path_utils import patch_gym_make
    patch_gym_make()
    patch_gym_make()  # idempotent
    import gymnasium as gym
    e = gym.make("Ant-v4")
    o, _ = e.reset(seed=0)
    assert o.shape == e.observation_space.shape
    e.close()


def test_eval_grid_iter_yields_all_combos():
    """grid_iter must produce one dict per cell, covering all combinations."""
    from src.eval import grid_iter
    grid = {"a": [1, 2, 3], "b": ["x", "y"]}
    cells = list(grid_iter(grid))
    assert len(cells) == 6
    keys = {c["a"] for c in cells}
    assert keys == {1, 2, 3}


def test_eval_perturbation_state_idempotent():
    """Apply the same perturbation twice and the model should be identical."""
    import gymnasium as gym
    import numpy as np
    from src.eval import PerturbationState

    env = gym.make("Ant-v4")
    state = PerturbationState(env)
    state.apply(1.2, 1.5, 0.9)
    bm_a = env.unwrapped.model.body_mass.copy()
    state.apply(1.2, 1.5, 0.9)
    bm_b = env.unwrapped.model.body_mass.copy()
    np.testing.assert_array_equal(bm_a, bm_b)
    env.close()


def test_eval_perturbation_state_restore():
    """After applying then restoring, defaults are preserved."""
    import gymnasium as gym
    import numpy as np
    from src.eval import PerturbationState

    env = gym.make("Ant-v4")
    state = PerturbationState(env)
    default_gear = env.unwrapped.model.actuator_gear.copy()
    state.apply(1.5, 2.0, 0.5)
    assert not np.array_equal(env.unwrapped.model.actuator_gear, default_gear)
    state.restore()
    np.testing.assert_array_equal(env.unwrapped.model.actuator_gear, default_gear)
    env.close()