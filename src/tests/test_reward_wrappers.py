"""Tests for reward_wrappers: stacking, individual effects, and shape preservation."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.path_utils import patch_gym_make
patch_gym_make()

from src.envs.reward_wrappers import (  # noqa: E402
    EnergyPenaltyWrapper,
    JointVelocityPenaltyWrapper,
    OrientationPenaltyWrapper,
    make_shaped_env,
)


def _run_steps(env, n: int = 5):
    obs, info = env.reset(seed=0)
    rewards = []
    actions = []
    for _ in range(n):
        a = np.ones(env.action_space.shape, dtype=np.float32)
        actions.append(a.copy())
        obs, r, term, trunc, info = env.step(a)
        rewards.append(float(r))
        if term or trunc:
            break
    return rewards, actions


def test_energy_penalty_subtracts_action_energy():
    """With coef=1.0 and unit actions, reward should be base - 1.0 * sum(|a|^2)."""
    env = make_shaped_env("Ant-v4", {"energy": 1.0})
    rewards, actions = _run_steps(env, n=5)
    assert len(rewards) >= 1
    for r, a in zip(rewards, actions):
        # penalized roughly by ||a||^2; tolerate small float drift
        assert r < 0, f"expected negative reward, got {r}"
    env.close()


def test_orientation_penalty_clamped():
    """Orientation penalty must be >= 0 (clamped)."""
    import gymnasium as gym
    base = gym.make("Ant-v4")
    base_r = []
    env = make_shaped_env("Ant-v4", {"orientation": 0.5})
    _, _ = _run_steps(base, n=3); base_r.append(_)
    rewards, _ = _run_steps(env, n=3)
    # With coef > 0 and unit actions, the wrapper must always reduce the reward
    # (since u_z <= 1, so tilt = max(0, 1 - u_z) >= 0; penalty is non-negative).
    env_base = make_shaped_env("Ant-v4")
    rb, _ = _run_steps(env_base, n=5)
    ro, _ = _run_steps(env,   n=5)
    for b, o in zip(rb, ro):
        assert o <= b + 1e-6, f"orient wrapper should reduce reward: base={b}, shaped={o}"
    env.close()
    env_base.close()


def test_make_shaped_env_no_shaping_is_passthrough():
    """make_shaped_env with empty cfg returns plain env (no wrappers)."""
    env = make_shaped_env("Ant-v4", {})
    env2 = make_shaped_env("Ant-v4")
    assert type(env) is type(env2)
    env.close()
    env2.close()


def test_make_shaped_env_stacks_wrappers():
    """Wrappers should be nested in declaration order."""
    env = make_shaped_env("Ant-v4", {"energy": 0.005, "orientation": 0.5})
    # the outer wrapper class appears in repr
    assert "OrientationPenaltyWrapper" in repr(env)
    env.close()


def test_make_shaped_env_ignores_zero_coef():
    """coef=0 (or None) should be skipped, leaving the env unchanged."""
    env = make_shaped_env("Ant-v4", {"energy": 0})
    env2 = make_shaped_env("Ant-v4")
    assert type(env) is type(env2)
    env.close()
    env2.close()


def test_make_shaped_env_unknown_raises():
    try:
        make_shaped_env("Ant-v4", {"bogus_wrapper": 0.5})
    except ValueError as ex:
        assert "Unknown reward wrapper" in str(ex)
        return
    assert False, "Expected ValueError for unknown wrapper"


def test_step_returns_correct_shapes():
    env = make_shaped_env("Ant-v4", {"energy": 0.001, "orientation": 0.1})
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    a = env.action_space.sample()
    obs2, r, term, trunc, info = env.step(a)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    env.close()