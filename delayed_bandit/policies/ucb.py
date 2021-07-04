import math
from typing import Tuple

import numpy as np
from numpy.random import Generator

from delayed_bandit.policies.policy import Policy


class UCB(Policy):
    def __init__(self, num_arms: int, alpha: float, rng: Generator):
        """
        Create Upper Confidence Bound policy. Based on Hoeffding’s inequality
        to build confidence intervals. alpha is to control the exploration-exploitation
        trade-off.
        """
        self._num_arms = num_arms
        self._alpha = alpha
        self._rng = rng
        self._current_arm = -1
        self.cumulative_rewards = np.zeros(num_arms, dtype=np.float32)
        self.arms_stats = np.zeros(num_arms, dtype=np.int32)

    def choice(self, t: int) -> int:
        if t < self._num_arms:
            self._current_arm = t
            return self._current_arm

        indexes = np.array([self._index(t=t, arm=i) for i in range(self._num_arms)])
        idx = np.where(indexes == np.max(indexes))
        return self._rng.choice(idx)

    def feed_reward(self, t: int, arm: int, reward: float):
        if arm != self._current_arm:
            raise ValueError(f"Expected the reward for arm {self._current_arm}, but got for {arm}")
        self.cumulative_rewards[arm] += reward
        self.arms_stats[arm] += 1
        return

    def empirically_best_arm(self) -> Tuple[int, float]:
        if np.count_nonzero(self.cumulative_rewards) == 0:
            return self._rng.choice(self._num_arms), 0.0
        idx = np.where(self.arms_stats != 0)
        i = np.argmax(self.cumulative_rewards[idx] / self.arms_stats[idx])
        arm = idx[0][i]
        return arm, self.cumulative_rewards[arm] / self.arms_stats[arm]

    def _index(self, t, arm) -> float:
        mean = self.cumulative_rewards[arm] / self.arms_stats[arm]
        confidence_radius = math.sqrt((self._alpha * math.log(t + 1)) / self.arms_stats[arm])
        return mean + confidence_radius
