from typing import Tuple

import numpy as np
from numpy.random import Generator

from delayed_bandit.policies.policy import Policy


class BetaThompsonSampling(Policy):
    def __init__(self, num_arms: int, rng: Generator):
        """
        Create Thompson Sampling policy with Beta distribution as a prior one.
        """
        self._num_arms = num_arms
        self._rng = rng
        self._current_arm = -1
        self.cumulative_rewards = np.zeros(num_arms, dtype=np.int32)
        self.arms_stats = np.zeros(num_arms, dtype=np.int32)

    def choice(self, t: int) -> int:
        samples = np.zeros(self._num_arms, dtype=np.float32)
        for a in range(self._num_arms):
            num_successes = self.cumulative_rewards[a]
            num_failures = self.arms_stats[a] - num_successes
            samples[a] = self._rng.beta(num_successes + 1, num_failures + 1)

        self._current_arm = int(np.argmax(samples))
        return self._current_arm

    def feed_reward(self, t: int, arm: int, reward: float):
        if arm != self._current_arm:
            raise ValueError(f"Expected the reward for arm {self._current_arm}, but got for {arm}")
        self.cumulative_rewards[arm] += round(reward)
        self.arms_stats[arm] += 1
        return

    def empirically_best_arm(self) -> Tuple[int, float]:
        idx = np.where(self.arms_stats != 0)
        i = np.argmax(self.cumulative_rewards[idx] / self.arms_stats[idx])
        arm = idx[0][i]
        return arm, self.cumulative_rewards[arm] / self.arms_stats[arm]
