"""Synthetic preference oracle using a noisy Peak-End heuristic."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from env_utils import PreferenceSample, TrajectorySegment


class SyntheticOracle:
    """Simulates noisy human preference labels using true trajectory rewards."""

    def __init__(
        self,
        peak_weight: float = 0.7,
        end_weight: float = 0.3,
        noise_prob: float = 0.10,
        seed: int = 0,
    ) -> None:
        if peak_weight + end_weight <= 0.0:
            raise ValueError("peak_weight + end_weight must be positive.")
        self.peak_weight = peak_weight
        self.end_weight = end_weight
        self.noise_prob = noise_prob
        self.rng = np.random.default_rng(seed)

    def utility(self, segment: TrajectorySegment) -> float:
        """Compute Peak-End utility from true rewards in one segment."""
        rewards = segment.true_rewards
        peak_reward = float(np.max(rewards))
        end_reward = float(rewards[-1])
        return self.peak_weight * peak_reward + self.end_weight * end_reward

    def label_pair(self, segment_a: TrajectorySegment, segment_b: TrajectorySegment) -> int:
        """Return binary preference label y=1 if A preferred over B, with noise."""
        utility_a = self.utility(segment_a)
        utility_b = self.utility(segment_b)
        label = 1 if utility_a > utility_b else 0

        if self.rng.random() < self.noise_prob:
            label = 1 - label
        return label

    def build_preference_samples(
        self,
        pairs: Iterable[Tuple[TrajectorySegment, TrajectorySegment]],
    ) -> List[PreferenceSample]:
        """Label segment pairs and return PreferenceSample objects."""
        samples: List[PreferenceSample] = []
        for segment_a, segment_b in pairs:
            label = self.label_pair(segment_a, segment_b)
            samples.append(PreferenceSample(segment_a=segment_a, segment_b=segment_b, label=label))
        return samples
