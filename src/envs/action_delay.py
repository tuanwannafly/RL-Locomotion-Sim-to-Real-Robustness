"""ActionDelayWrapper: queue actions so that the action applied at step t
is the one the policy produced at step ``t - delay - 1``.

This is a standard sim-to-real trick to simulate the control-loop latency of
real robots. *delay* is in steps (assumed at the env's control frequency).
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class ActionDelayWrapper(gym.Wrapper):
    """Apply the policy's action with a fixed-step delay.

    The policy still receives the same observation cadence; only the action
    applied to the simulator is delayed.
    """

    def __init__(self, env: gym.Env, delay: int = 1):
        super().__init__(env)
        if delay < 0:
            raise ValueError("delay must be >= 0")
        self.delay = int(delay)
        self._queue: deque = deque(maxlen=self.delay + 1)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # flush queue
        self._queue.clear()
        return out

    def step(self, action):
        self._queue.append(np.asarray(action, dtype=np.float32))
        if len(self._queue) <= self.delay:
            # Not enough history yet: hold the env's "default" action (zeros).
            applied = np.zeros_like(self._queue[0])
        else:
            applied = self._queue[0]  # oldest queued action
        obs, reward, terminated, truncated, info = self.env.step(applied)
        info["action_delay_applied"] = applied
        return obs, reward, terminated, truncated, info