"""Domain Randomization wrapper for MuJoCo locomotion envs.

Randomizes the following physical parameters at every ``reset()``:

* body mass of every non-floor body (``+/- body_mass_range``)
* joint friction loss (``friction_range`` * default)
* motor actuator gear / force scale (``+/- actuator_gain_range``)

Plus stochastic observation noise (Gaussian on every obs component) and
an optional action delay.

The wrapper is *physics-only*: the action and observation spaces stay the
same so any SB3 policy can train with or without the wrapper.
"""
from __future__ import annotations

from collections import deque
from typing import Mapping

import gymnasium as gym
import numpy as np

from src.envs.action_delay import ActionDelayWrapper


class DomainRandomizationWrapper(gym.Wrapper):
    """Randomize MuJoCo physics params at reset; optionally delay actions and
    add Gaussian obs noise."""

    def __init__(
        self,
        env: gym.Env,
        body_mass_range: tuple[float, float] = (0.8, 1.2),
        friction_range: tuple[float, float] = (0.5, 1.5),
        actuator_gain_range: tuple[float, float] = (0.8, 1.2),
        obs_noise_std: float = 0.05,
        action_delay_steps: int = 0,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(env)
        self.body_mass_range = body_mass_range
        self.friction_range = friction_range
        self.actuator_gain_range = actuator_gain_range
        self.obs_noise_std = float(obs_noise_std)
        self.action_delay_steps = int(action_delay_steps)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Snapshot defaults so we can multiply instead of re-assigning absolute values.
        model = self.env.unwrapped.model
        self._default_body_mass = model.body_mass.copy()
        # Joint friction loss lives at the DOF level (`dof_frictionloss`, shape = nv).
        self._default_dof_frictionloss = model.dof_frictionloss.copy()
        # Motor strength lives in `actuator_gear` (gear ratio * sign, shape = (nu, 6)).
        # For motor actuators (trntype=0), only the first column is used.
        self._default_actuator_gear = model.actuator_gear.copy()

        # Defer action-delay handling to a sub-wrapper so the action logic is isolated.
        if self.action_delay_steps > 0:
            # We will apply the delay ourselves inside step() instead of wrapping, to keep
            # reset() able to inspect the env directly.
            self._delay_queue: deque = deque(maxlen=self.action_delay_steps + 1)

    # ---- randomization helpers ----

    def _randomize_body_mass(self) -> None:
        model = self.env.unwrapped.model
        n_bodies = model.nbody
        if n_bodies <= 1:
            return
        # Skip body 0 (world) and body 1 (root, often floor); randomize all others.
        for b in range(2, n_bodies):
            scale = self.rng.uniform(*self.body_mass_range)
            model.body_mass[b] = self._default_body_mass[b] * scale

    def _randomize_friction(self) -> None:
        model = self.env.unwrapped.model
        scale = self.rng.uniform(*self.friction_range)
        model.dof_frictionloss[:] = self._default_dof_frictionloss * scale

    def _randomize_actuator_gain(self) -> None:
        model = self.env.unwrapped.model
        for a in range(model.nu):
            scale = self.rng.uniform(*self.actuator_gain_range)
            model.actuator_gear[a, 0] = self._default_actuator_gear[a, 0] * scale

    def reset(self, **kwargs):
        # Restore defaults before re-randomizing so each episode is independent.
        model = self.env.unwrapped.model
        model.body_mass[:] = self._default_body_mass
        model.dof_frictionloss[:] = self._default_dof_frictionloss
        model.actuator_gear[:] = self._default_actuator_gear

        self._randomize_body_mass()
        self._randomize_friction()
        self._randomize_actuator_gain()

        if self.action_delay_steps > 0:
            self._delay_queue.clear()

        obs, info = self.env.reset(**kwargs)
        obs = self._add_obs_noise(obs)
        return obs, info

    def _add_obs_noise(self, obs):
        if self.obs_noise_std <= 0:
            return obs
        noise = self.rng.normal(0.0, self.obs_noise_std, size=np.asarray(obs).shape)
        return np.asarray(obs, dtype=np.float32) + noise.astype(np.float32)

    def step(self, action):
        if self.action_delay_steps > 0:
            self._delay_queue.append(np.asarray(action, dtype=np.float32))
            if len(self._delay_queue) <= self.action_delay_steps:
                applied = np.zeros(self.env.action_space.shape, dtype=np.float32)
            else:
                applied = self._delay_queue[0]
            obs, reward, terminated, truncated, info = self.env.step(applied)
            info = dict(info)
            info["action_delay_applied"] = applied
            info["action_delay_steps"] = self.action_delay_steps
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._add_obs_noise(obs)
        return obs, reward, terminated, truncated, info


def make_dr_env(env_id: str, dr_cfg: Mapping | None, seed: int = 0,
                shaping_cfg: Mapping | None = None) -> gym.Env:
    """Convenience builder: make base env + apply DR + apply reward shaping.

    *dr_cfg* keys: body_mass, friction, actuator_gain, obs_noise_std, action_delay_steps.
    Values are dicts with "low"/"high" (or scalars for noise/delay).
    """
    # local imports to avoid circular
    from src.envs.reward_wrappers import make_shaped_env
    env = make_shaped_env(env_id, shaping_cfg=shaping_cfg)
    if not dr_cfg or not dr_cfg.get("enabled", False):
        return env
    rng = np.random.default_rng(seed)
    bm = _to_range(dr_cfg.get("body_mass", (0.8, 1.2)))
    fr = _to_range(dr_cfg.get("friction", (0.5, 1.5)))
    ag = _to_range(dr_cfg.get("actuator_gain", (0.8, 1.2)))
    noise = float(dr_cfg.get("obs_noise_std", 0.05))
    delay_cfg = dr_cfg.get("action_delay_steps", 0)
    if isinstance(delay_cfg, Mapping):
        delay = int(rng.integers(int(delay_cfg.get("low", 0)), int(delay_cfg.get("high", 0)) + 1))
    else:
        delay = int(delay_cfg)
    return DomainRandomizationWrapper(
        env,
        body_mass_range=bm,
        friction_range=fr,
        actuator_gain_range=ag,
        obs_noise_std=noise,
        action_delay_steps=delay,
        rng=rng,
    )


def _to_range(spec) -> tuple[float, float]:
    """Coerce a (low, high) tuple or {'low': .., 'high': ..} dict into a tuple."""
    if isinstance(spec, Mapping):
        return (float(spec.get("low", 0.8)), float(spec.get("high", 1.2)))
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return (float(spec[0]), float(spec[1]))
    raise ValueError(f"Cannot interpret range spec: {spec!r}")