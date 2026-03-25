"""Environment and trajectory utilities for memory-constrained PbRL."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class TrajectorySegment:
    """Fixed-length trajectory segment used for preference comparisons."""

    states: np.ndarray
    actions: np.ndarray
    true_rewards: np.ndarray


@dataclass(frozen=True)
class PreferenceSample:
    """A labeled preference sample between two trajectory segments."""

    segment_a: TrajectorySegment
    segment_b: TrajectorySegment
    label: int


class MaskTrueRewardWrapper(gym.Wrapper):
    """Gym wrapper that hides true reward from the agent and exposes it via info."""

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        next_state, true_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["true_reward"] = float(true_reward)
        masked_reward = 0.0
        return next_state, masked_reward, terminated, truncated, info


class ContinuousGridworldEnv(gym.Env[np.ndarray, np.ndarray]):
    """Lightweight continuous 2D control task used when Box2D is unavailable.

    State: [x, y, vx, vy]
    Action: [ax, ay] in [-1, 1]
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 300, dt: float = 0.1) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.dt = dt
        self.bounds = 2.0
        self.goal = np.array([1.0, 1.0], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-self.bounds, -self.bounds, -3.0, -3.0], dtype=np.float32),
            high=np.array([self.bounds, self.bounds, 3.0, 3.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._state = np.zeros(4, dtype=np.float32)
        self._steps = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._steps = 0

        pos = self.np_random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)
        vel = np.zeros(2, dtype=np.float32)
        self._state = np.concatenate([pos, vel], axis=0).astype(np.float32)
        return self._state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        pos = self._state[:2]
        vel = self._state[2:]

        vel = np.clip(vel + action * self.dt, -3.0, 3.0)
        pos = np.clip(pos + vel * self.dt, -self.bounds, self.bounds)
        self._state = np.concatenate([pos, vel], axis=0).astype(np.float32)
        self._steps += 1

        dist = float(np.linalg.norm(pos - self.goal))
        action_penalty = 0.05 * float(np.square(action).sum())
        true_reward = -dist - action_penalty

        reached_goal = dist < 0.1
        terminated = reached_goal
        truncated = self._steps >= self.max_steps

        if reached_goal:
            true_reward += 5.0

        return self._state.copy(), true_reward, terminated, truncated, {}


class SegmentExtractor:
    """Collects strict non-overlapping fixed-length segments from rollouts."""

    def __init__(self, segment_length: int) -> None:
        self.segment_length = segment_length
        self._states: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []

    def add_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        true_reward: float,
    ) -> Optional[TrajectorySegment]:
        """Add one transition and return a segment once K steps are accumulated."""
        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32))
        self._rewards.append(float(true_reward))

        if len(self._states) < self.segment_length:
            return None

        segment = TrajectorySegment(
            states=np.stack(self._states, axis=0),
            actions=np.stack(self._actions, axis=0),
            true_rewards=np.asarray(self._rewards, dtype=np.float32),
        )
        self.reset()
        return segment

    def reset(self) -> None:
        """Clear in-progress segment buffer."""
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()


def make_lunar_lander_env(seed: int) -> gym.Env:
    """Create LunarLanderContinuous-v2 environment with true reward masking."""
    env = gym.make("LunarLanderContinuous-v2")
    env = MaskTrueRewardWrapper(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_env(env_name: str, seed: int) -> gym.Env:
    """Create requested environment and always return reward-masked wrapper.

    Supported:
    - lunar_lander_continuous
    - continuous_gridworld
    - auto (try LunarLander then fallback to Gridworld)
    """
    normalized = env_name.strip().lower()

    if normalized in {"auto", "lunar_lander_continuous", "lunarlandercontinuous-v2"}:
        try:
            return make_lunar_lander_env(seed=seed)
        except Exception as exc:
            if normalized != "auto":
                raise RuntimeError(
                    "Failed to create LunarLanderContinuous-v2. "
                    "Install Box2D or use env_name='continuous_gridworld'."
                ) from exc
            env = ContinuousGridworldEnv()
            env = MaskTrueRewardWrapper(env)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

    if normalized in {"continuous_gridworld", "gridworld"}:
        env = ContinuousGridworldEnv()
        env = MaskTrueRewardWrapper(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    raise ValueError(f"Unknown env_name='{env_name}'.")


def sample_segment_pairs(
    segments: Sequence[TrajectorySegment],
    n_pairs: int,
    rng: np.random.Generator,
) -> List[Tuple[TrajectorySegment, TrajectorySegment]]:
    """Sample random segment pairs for active preference querying."""
    if len(segments) < 2:
        return []

    pairs: List[Tuple[TrajectorySegment, TrajectorySegment]] = []
    for _ in range(n_pairs):
        idx_a, idx_b = rng.choice(len(segments), size=2, replace=False)
        pairs.append((segments[idx_a], segments[idx_b]))
    return pairs


def set_global_seeds(seed: int) -> None:
    """Seed numpy and Python hash randomness where applicable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch may be unavailable in lightweight environments.
        pass
