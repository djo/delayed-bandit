from typing import Tuple

import numpy as np
from numpy.random import Generator

from delayed_bandit.policies.policy import Policy


class EpsilonGreedy(Policy):
    def __init__(self, num_arms: int, epsilon: float, rng: Generator):
        """
        Create EpsilonGreed policy playing the empirically best arm
        with probability (1 - epsilon), otherwise explores uniformly at random.
        """
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError(f"Epsilon {epsilon} must be in [0.0, 1.0]")
        self._num_arms = num_arms
        self._epsilon = epsilon
        self._rng = rng
        self._current_arm = -1
        self.cumulative_rewards = np.zeros(num_arms, dtype=np.float32)
        self.arms_stats = np.zeros(num_arms, dtype=np.int32)

    def choice(self, t: int) -> int:
        if self._rng.random() < self._epsilon:
            self._current_arm = self._rng.choice(self._num_arms)
            return self._current_arm
        self._current_arm, _ = self.empirically_best_arm()
        return self._current_arm

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

    def name(self) -> str:
        return f"Epsilon-Greedy(epsilon={self._epsilon})"
