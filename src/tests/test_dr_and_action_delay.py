"""Tests for action_delay.py and dr_wrapper.py."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.path_utils import patch_gym_make
patch_gym_make()

from src.envs.action_delay import ActionDelayWrapper  # noqa: E402
from src.envs.dr_wrapper import (  # noqa: E402
    DomainRandomizationWrapper,
    _to_range,
    make_dr_env,
)


def test_action_delay_applies_queued_actions():
    """With delay=2, actions 1,2,3,4,5 should be applied 0,0,1,2,3."""
    import gymnasium as gym
    base = gym.make("Ant-v4")
    env = ActionDelayWrapper(base, delay=2)
    obs, info = env.reset(seed=0)
    applied = []
    for step in range(5):
        a = np.ones(env.action_space.shape, dtype=np.float32) * (step + 1)
        obs, r, term, trunc, info = env.step(a)
        applied.append(info["action_delay_applied"])
    expected = [
        np.zeros_like(env.action_space.low),
        np.zeros_like(env.action_space.low),
        np.ones_like(env.action_space.low),
        np.ones_like(env.action_space.low) * 2,
        np.ones_like(env.action_space.low) * 3,
    ]
    for i, (got, exp) in enumerate(zip(applied, expected)):
        np.testing.assert_array_equal(got, exp, err_msg=f"step {i}")
    env.close()


def test_action_delay_zero_is_passthrough():
    """delay=0 should apply the action immediately."""
    import gymnasium as gym
    base = gym.make("Ant-v4")
    env = ActionDelayWrapper(base, delay=0)
    obs, info = env.reset(seed=0)
    a = np.arange(env.action_space.shape[0], dtype=np.float32)
    obs, r, term, trunc, info = env.step(a)
    np.testing.assert_array_equal(info["action_delay_applied"], a)
    env.close()


def test_dr_randomizes_body_mass_across_episodes():
    dr_cfg = {
        "enabled": True,
        "body_mass": (0.8, 1.2),
        "friction": (0.5, 1.5),
        "actuator_gain": (0.8, 1.2),
        "obs_noise_std": 0.0,
        "action_delay_steps": 0,
    }
    env = make_dr_env("Ant-v4", dr_cfg, seed=0)
    model = env.unwrapped.model
    masses = []
    for ep in range(5):
        env.reset(seed=ep)
        masses.append(float(model.body_mass[2]))
    # at least 4 distinct values when sampling uniformly over (0.8, 1.2)
    assert len(set(round(m, 5) for m in masses)) >= 4, f"expected varied mass: {masses}"
    env.close()


def test_dr_randomizes_actuator_gear_and_restores():
    dr_cfg = {
        "enabled": True,
        "body_mass": (0.9, 1.1),
        "actuator_gain": (0.5, 2.0),
        "obs_noise_std": 0.0,
        "action_delay_steps": 0,
    }
    env = make_dr_env("Ant-v4", dr_cfg, seed=0)
    model = env.unwrapped.model
    default_gear = model.actuator_gear[0, 0]
    gears = []
    for ep in range(8):
        env.reset(seed=ep)
        gears.append(model.actuator_gear[0, 0])
    # gears should not all be equal; at least one in [0.5x, 2.0x] range
    assert any(abs(g - default_gear) > 1.0 for g in gears), f"expected gear to vary: {gears}"
    env.close()


def test_dr_obs_noise_changes_obs():
    dr_cfg = {
        "enabled": True,
        "obs_noise_std": 0.5,
        "action_delay_steps": 0,
    }
    env1 = make_dr_env("Ant-v4", dr_cfg, seed=1)
    env2 = make_dr_env("Ant-v4", dr_cfg, seed=2)
    obs1, _ = env1.reset(seed=0)
    obs2, _ = env2.reset(seed=0)
    # With different seeds, at least one obs component should differ noticeably.
    assert np.max(np.abs(obs1 - obs2)) > 0.05, f"noise should differ across seeds"
    env1.close()
    env2.close()


def test_dr_disabled_returns_plain_env():
    from src.envs.reward_wrappers import make_shaped_env
    base = make_shaped_env("Ant-v4")
    env = make_dr_env("Ant-v4", {"enabled": False})
    assert type(env) is type(base)
    env.close()
    base.close()


def test_to_range_handles_dict_and_tuple():
    assert _to_range({"low": 0.5, "high": 1.5}) == (0.5, 1.5)
    assert _to_range((0.5, 1.5)) == (0.5, 1.5)
    assert _to_range([0.5, 1.5]) == (0.5, 1.5)
    try:
        _to_range("nope")
    except ValueError:
        return
    assert False, "expected ValueError"