"""Reward-shaping wrappers for MuJoCo locomotion envs.

Each wrapper adds a term to the reward computed by the underlying env. The
original reward is preserved; we *add* shaping terms (so the agent still
gets the dense forward-velocity signal that Ant-v4 / Walker2d-v4 provide).

Available wrappers:

* :class:`EnergyPenaltyWrapper` -- penalise large actions (sum of |a|^2).
* :class:`OrientationPenaltyWrapper` -- penalise torso tilt away from upright.
* :class:`JointVelocityPenaltyWrapper` -- penalise large joint velocities.
* :class:`ActionSmoothnessWrapper` -- penalise |a_t - a_{t-1}|^2.

Wrappers are stacked via :func:`make_shaped_env`.
"""
from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np


class EnergyPenaltyWrapper(gym.RewardWrapper):
    """Subtract a coefficient * sum(|a|^2) from the reward."""

    def __init__(self, env: gym.Env, coef: float = 0.005):
        super().__init__(env)
        self.coef = coef

    def reward(self, reward: float) -> float:
        # The action that was just taken is the most recent attribute on the env.
        action = getattr(self.env, "last_action", None)
        if action is None:
            # Older gymnasium versions don't expose last_action; fall back to current step.
            action = self.env.unwrapped.data.ctrl[:] if hasattr(self.env, "unwrapped") else None
        if action is None:
            return reward
        return float(reward - self.coef * float(np.sum(np.square(action))))


class OrientationPenaltyWrapper(gym.RewardWrapper):
    """Penalise torso pitch + roll away from upright (z-axis world frame).

    Works for any MuJoCo env where the root body is index 1 (Ant / Walker2d
    both follow that convention). The penalty is 1 - cos(angle between
    torso up-axis and world up-axis), so it is in [0, 1].
    """

    def __init__(self, env: gym.Env, coef: float = 0.5):
        super().__init__(env)
        self.coef = coef

    def reward(self, reward: float) -> float:
        try:
            data = self.env.unwrapped.data
            # torso up axis (body z direction, world frame) for the root body
            torso_up = data.xmat[1, 2]  # z-component of xmat row 1
            # we want z close to +1 (upright); deviation = 1 - torso_up, clamped >= 0
            tilt = float(max(0.0, 1.0 - torso_up))
            return float(reward - self.coef * tilt)
        except (AttributeError, IndexError):
            return reward


class JointVelocityPenaltyWrapper(gym.RewardWrapper):
    """Penalise large joint velocities (sum of |qvel|^2)."""

    def __init__(self, env: gym.Env, coef: float = 0.001):
        super().__init__(env)
        self.coef = coef

    def reward(self, reward: float) -> float:
        try:
            qvel = self.env.unwrapped.data.qvel
            return float(reward - self.coef * float(np.sum(np.square(qvel))))
        except AttributeError:
            return reward


class ActionSmoothnessWrapper(gym.RewardWrapper):
    """Penalise |a_t - a_{t-1}|^2. Requires a custom step to track prev action."""

    def __init__(self, env: gym.Env, coef: float = 0.01):
        super().__init__(env)
        self.coef = coef
        self._prev_action: np.ndarray | None = None

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._prev_action = None
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action = np.asarray(action, dtype=np.float32)
        if self._prev_action is not None and self.coef > 0:
            smooth_pen = float(np.sum(np.square(action - self._prev_action)))
            reward = float(reward - self.coef * smooth_pen)
            info["smoothness_penalty"] = smooth_pen
        self._prev_action = action
        return obs, reward, terminated, truncated, info

    def reward(self, reward):
        # RewardWrapper expects this method; we handle the penalty inside step().
        return reward


# Registry mapping string name -> wrapper class
REWARD_WRAPPERS: dict[str, Callable[..., gym.RewardWrapper]] = {
    "energy": EnergyPenaltyWrapper,
    "orientation": OrientationPenaltyWrapper,
    "joint_vel": JointVelocityPenaltyWrapper,
    "smoothness": ActionSmoothnessWrapper,
}


def make_shaped_env(env_id: str, shaping_cfg: dict | None = None, **env_kwargs) -> gym.Env:
    """Build an env with the requested reward wrappers stacked.

    *shaping_cfg* is a dict like::

        {"energy": 0.005, "orientation": 0.5}
    """
    env = gym.make(env_id, **env_kwargs)
    if not shaping_cfg:
        return env
    for name, coef in shaping_cfg.items():
        if coef is None or coef == 0:
            continue
        wrapper_cls = REWARD_WRAPPERS.get(name)
        if wrapper_cls is None:
            raise ValueError(f"Unknown reward wrapper {name!r}; choices: {list(REWARD_WRAPPERS)}")
        env = wrapper_cls(env, coef=coef)
    return env